
from langdetect import detect
import whisper

from moviepy import VideoFileClip
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from dotenv import load_dotenv

import PyPDF2
import docx
import tempfile

import torch

from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import io
import json
from PIL import Image
from flask import Flask, request, jsonify, send_file
import cv2
import uuid
from deepface import DeepFace
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import os
import google.generativeai as genai
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin
# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Determine transcript file
translated_file = "translated_transcript.txt" if os.path.exists("translated_transcript.txt") else "translated.txt"
# --- Setup OpenAI Client for LLaMA 3 ---
client = OpenAI(
    base_url="https://infer.e2enetworks.net/project/p-5454/genai/llama_3_3_70b_instruct_fp8/v1",
    api_key=os.getenv("E2E_API_KEY")  # Or paste your API Key directly here (not recommended)
)

# --- Load Whisper Model for optional transcription ---
model = whisper.load_model("small", device="cuda")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///usersdata.db'
app.config['SECRET_KEY'] = 'fdsfasdfsad34234sdfsd'

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Language options
LANG_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur"
}

# Load Whisper Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



login_manager = LoginManager()

# Initialize your login manager
login_manager.init_app(app)

def read_transcript():
    """Read the translated transcript."""
    if os.path.exists(translated_file):
        with open(translated_file, "r", encoding="utf-8") as file:
            return file.read()
    else:
        st.error("No translated transcript file found!")
        return None


# Define the user_loader function
@login_manager.user_loader
def load_user(user_id):
    # Assuming you have a User model with an 'id' field
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


class EmotionAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    top_emotion = db.Column(db.String(50))
    second_emotion = db.Column(db.String(50))
    distress_percentage = db.Column(db.Float)
    alert_triggered = db.Column(db.Boolean)
    chart_image = db.Column(db.LargeBinary)
    pie_chart_analysis = db.Column(db.String(1000))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    student = db.relationship('User', backref=db.backref('analysis', uselist=False))


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()

    # Validate input
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing username or password'}), 400

    username = data.get('username')
    password = data.get('password')

    # Check if the username already exists
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'error': 'Username already exists'}), 400

    # Create a new user
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    # Validate input
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing username or password'}), 400

    username = data.get('username')
    password = data.get('password')

    # Find the user by username
    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid username or password'}), 401

    # Login the user
    login_user(user)

    return jsonify({'message': 'Login successful', 'student_id': user.id, 'username': username}), 200


@app.route('/api/logout', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def logout():
    # Log out the current user
    logout_user()

    return jsonify({"message": "Logout successful"}), 200


@app.route('/api/analyze-emotion', methods=['POST'])
def analyze_emotion():
    if 'video' not in request.files or 'student_id' not in request.form:
        return jsonify({'error': 'Missing video file or student ID'}), 400

    video_file = request.files['video']
    student_id = request.form['student_id']

    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400

    # Save video temporarily
    temp_video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
    video_file.save(temp_video_path)

    try:
        # Process and save analysis to DB
        result = process_video(temp_video_path, student_id)

        return jsonify({
            'top_emotion': result['top_emotion'],
            'second_emotion': result['second_emotion'],
            'distress_percentage': result['distress_percentage'],
            'alert_triggered': result['alert_triggered']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@app.route('/api/students/all', methods=['GET'])
def get_all_students():
    """Get all students' details along with their emotion analysis data"""
    try:
        # Query all students with their analysis data using the relationship
        students = User.query.all()

        students_data = []
        for student in students:
            # Get the emotion analysis data for this student
            analysis = student.analysis  # This uses the backref from the relationship

            student_info = {
                'id': student.id,
                'username': student.username,
                'analysis': None
            }

            if analysis:
                # Convert binary chart image to base64 for frontend display
                chart_base64 = None
                if analysis.chart_image:
                    import base64
                    chart_base64 = base64.b64encode(analysis.chart_image).decode('utf-8')

                student_info['analysis'] = {
                    'top_emotion': analysis.top_emotion,
                    'second_emotion': analysis.second_emotion,
                    'distress_percentage': analysis.distress_percentage,
                    'alert_triggered': analysis.alert_triggered,
                    'chart_image_base64': chart_base64,  # Base64 encoded image
                    'timestamp': analysis.timestamp.isoformat() if analysis.timestamp else None,
                    'pie_chart_analysis': analysis.pie_chart_analysis
                }

            students_data.append(student_info)

        return jsonify({
            'students': students_data,
            'total_count': len(students_data),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to fetch students data: {str(e)}'}), 500


@app.route('/api/download-chart', methods=['GET'])
def download_chart():
    return send_file(os.path.join(RESULTS_FOLDER, "chart.png"), as_attachment=True)


@app.route('/api/download-report', methods=['GET'])
def download_report():
    return send_file(os.path.join(RESULTS_FOLDER, "emotion_report.txt"), as_attachment=True)


def process_video(video_path, student_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    emotion_counts = Counter()
    total_frames = 0
    alert_emotions = ["sad", "angry", "fear"]
    distress_threshold = 15
    frame_skip = 30
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi = rgb[y:y + h, x:x + w]
                try:
                    result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    emotion_counts[emotion] += 1
                except Exception as e:
                    print(f"Emotion detection error: {e}")

            total_frames += 1
        frame_count += 1

    cap.release()

    # Results
    top_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "None"
    second_emotion = emotion_counts.most_common(2)[1][0] if len(emotion_counts) > 1 else "None"

    distress_frames = sum(emotion_counts[e] for e in alert_emotions if e in emotion_counts)
    if total_frames == 0:
        raise ValueError("No valid frames were processed. Emotion detection may have failed.")
    distress_percentage = (distress_frames / total_frames) * 100
    alert_triggered = distress_percentage >= distress_threshold

    # Generate pie chart
    chart_stream = io.BytesIO()
    plt.figure(figsize=(6, 6))
    if total_frames > 0 and emotion_counts:
        labels = emotion_counts.keys()
        sizes = [emotion_counts[e] for e in labels]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title("Emotion Distribution Over Time")
    else:
        plt.text(0.5, 0.5, "No emotions detected", ha='center', va='center', fontsize=12)
        plt.title("No Data Available")

    # Save chart to disk
    chart_filename = f"chart_{student_id}.png"
    chart_path = os.path.join(RESULTS_FOLDER, chart_filename)
    plt.savefig(chart_path)

    # Also save chart to memory for DB
    chart_stream = io.BytesIO()
    plt.savefig(chart_stream, format='png')
    plt.close()
    chart_stream.seek(0)
    resultAnalysis = analyze_piechart(chart_filename)
    print(resultAnalysis)
    # Save or update in DB
    existing = EmotionAnalysis.query.filter_by(student_id=student_id).first()
    if existing:
        existing.top_emotion = top_emotion
        existing.second_emotion = second_emotion
        existing.distress_percentage = round(distress_percentage, 2)
        existing.alert_triggered = alert_triggered
        existing.chart_image = chart_stream.read()
        existing.timestamp = datetime.utcnow()
        existing.pie_chart_analysis = resultAnalysis
    else:
        new_record = EmotionAnalysis(
            student_id=student_id,
            top_emotion=top_emotion,
            second_emotion=second_emotion,
            distress_percentage=round(distress_percentage, 2),
            alert_triggered=alert_triggered,
            chart_image=chart_stream.read(),
            pie_chart_analysis=resultAnalysis
        )
        db.session.add(new_record)

    db.session.commit()

    return {
        'top_emotion': top_emotion,
        'second_emotion': second_emotion,
        'distress_percentage': round(distress_percentage, 2),
        'alert_triggered': alert_triggered
    }


#emotion extraction code
def extract_emotions_from_pie_chart(image_path):
    """Uses Gemini to extract text (emotion percentages) from the pie chart image."""
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Open the image using PIL
    image = Image.open(image_path)

    prompt = """Extract emotion percentages from this pie chart image.
    The emotions include: neutral, happy, sad, angry, fear, and disgust.
    Provide the output in JSON format like this:
    {
        "neutral": 60,
        "happy": 30,
        "sad": 5,
        "angry": 3,
        "fear": 2,
        "disgust": 0
    }
    IMPORTANT: Only respond with the JSON data and nothing else."""

    response = model.generate_content([prompt, image])

    try:
        # Print the raw response to help with debugging

        # Extract the JSON part from the response
        # First try to parse directly
        try:
            emotion_data = json.loads(response.text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to find and extract JSON content
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                emotion_data = json.loads(json_str)
            else:
                raise Exception("Could not find valid JSON in the response")

        return Counter(emotion_data)
    except Exception as e:
        print("Error parsing OCR output:", e)
        print("Full response was:", response.text)
        return None


def analyze_emotion_data(emotion_counts):
    """Analyzes emotion data to determine distress levels."""
    total_frames = sum(emotion_counts.values())
    distress_emotions = ["sad", "angry", "fear", "disgust"]

    distress_percentage = sum(emotion_counts.get(e, 0) for e in distress_emotions) / total_frames * 100
    neutral_happy_percentage = (emotion_counts.get("neutral", 0) + emotion_counts.get("happy", 0)) / total_frames * 100

    return distress_percentage, neutral_happy_percentage


def generate_teacher_feedback(distress_percentage):
    """Generates personalized, structured feedback using Gemini based on emotional distress levels."""
    if distress_percentage >= 20:
        prompt = f"""
        A student has shown a high emotional distress level of {distress_percentage:.2f}% in the classroom.
        Do not mention any intro line such as Here's structured feedback based on the provided information just provide 
        the response.Generate structured and personalized feedback in plain text format (no markdown or bullet symbols).
        Include the following 4 labeled sections with exactly 2 points each:

        Personalized Feedback
        - A brief summary of the student’s emotional state and classroom engagement.

        Strengths
        - Two observed strengths based on emotional behavior.

        Areas for Growth
        - Two specific areas where the student may need support or improvement.

        Recommended Actions
        - Two practical and supportive actions the teacher can take to help the student.
        """

    elif distress_percentage >= 10:
        prompt = f"""
        A student has shown a moderate emotional distress level of {distress_percentage:.2f}% in the classroom. Do 
        not mention any intro line such as Here's structured feedback based on the provided information just provide 
        the response. Generate personalized, plain text feedback with the following 4 labeled sections and exactly 2 
        points in each:

        Personalized Feedback
        - A brief overview of the student’s emotional state and general behavior.

        Strengths
        - Two positive aspects of the student’s emotional performance.

        Areas for Growth
        - Two areas where the student could benefit from improvement.

        Recommended Actions
        - Two helpful and practical suggestions for the teacher to support the student further.
        """

    else:
        return (
            "Personalized Feedback\n"
            "The student shows excellent emotional balance and positive classroom behavior.\n"
            "They are likely comfortable and actively participating.\n\n"
            "Strengths\n"
            "Consistent emotional stability.\n"
            "Shows regular engagement and attention.\n\n"
            "Areas for Growth\n"
            "Could benefit from occasional new learning challenges.\n"
            "May enjoy more variety in learning activities.\n\n"
            "Recommended Actions\n"
            "Continue using interactive teaching methods.\n"
            "Incorporate occasional student-led activities."
        )

    # Generate structured feedback using Gemini
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def analyze_piechart(chartname):
    """Analyze the pie chart and return teacher feedback as a string"""
    chart_path = os.path.join(RESULTS_FOLDER, chartname)

    # Check if chart exists
    if not os.path.exists(chart_path):
        return "No chart available. Please run /analyze-emotion first."

    try:
        # Extract emotion data from the pie chart
        emotion_counts = extract_emotions_from_pie_chart(chart_path)
        if not emotion_counts:
            return "Error extracting emotions from pie chart."
        # Analyze extracted emotion data
        distress_percentage, neutral_happy_percentage = analyze_emotion_data(emotion_counts)
        # Generate feedback for teachers
        feedback = generate_teacher_feedback(distress_percentage)
        # Return feedback as plain string
        return feedback

    except Exception as e:
        return f"An error occurred: {str(e)}"


def parse_feedback_sections(feedback_text):
    sections = {
        "personalized_feedback": "",
        "strengths": "",
        "areas_for_growth": "",
        "recommended_actions": ""
    }

    # Define section headers and their respective keys
    section_map = {
        "Personalized Feedback": "personalized_feedback",
        "Strengths": "strengths",
        "Areas for Growth": "areas_for_growth",
        "Recommended Actions": "recommended_actions"
    }

    current_section = None
    lines = feedback_text.strip().splitlines()
    for line in lines:
        header = line.strip().rstrip(":")
        if header in section_map:
            current_section = section_map[header]
            continue
        elif current_section:
            sections[current_section] += line + "\n"

    # Strip trailing whitespace from each section
    for key in sections:
        sections[key] = sections[key].strip()

    return sections


@app.route('/api/analyze-pie', methods=['GET'])
def analyze_pie():
    """Analyze the pie chart and return emotion analysis results"""
    chart_path = os.path.join(RESULTS_FOLDER, "chart.png")

    # Check if chart exists
    if not os.path.exists(chart_path):
        return jsonify({"error": "No chart available. Please run /analyze-emotion first."}), 400

    try:
        # Extract emotion data from the pie chart
        emotion_counts = extract_emotions_from_pie_chart(chart_path)
        if not emotion_counts:
            return jsonify({"error": "Error extracting emotions from pie chart."}), 500

        # Analyze extracted emotion data
        distress_percentage, neutral_happy_percentage = analyze_emotion_data(emotion_counts)

        # Generate feedback for teachers
        feedback = generate_teacher_feedback(distress_percentage)

        # Save feedback to file
        feedback_path = os.path.join(RESULTS_FOLDER, "teacher_feedback.txt")
        with open(feedback_path, "w", encoding="utf-8") as file:
            file.write(feedback)

        # Return JSON response with analysis results
        return jsonify({
            "emotion_counts": dict(emotion_counts),
            "distress_percentage": distress_percentage,
            "neutral_happy_percentage": neutral_happy_percentage,
            "feedback": feedback,
            "feedback_download_url": "/api/download/feedback"
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/api/download/feedback', methods=['GET'])
def download_feedback():
    """Download the teacher feedback as a text file"""
    feedback_path = os.path.join(RESULTS_FOLDER, "teacher_feedback.txt")

    # Check if feedback file exists
    if not os.path.exists(feedback_path):
        return jsonify({"error": "No feedback available. Please run /analyze-pie first."}), 400

    try:
        return send_file(
            feedback_path,
            as_attachment=True,
            download_name="teacher_feedback.txt",
            mimetype="text/plain"
        )
    except Exception as e:
        return jsonify({"error": f"Error downloading feedback: {str(e)}"}), 500


@app.route('/api/upload-chart', methods=['POST'])
def upload_chart():
    """Upload a pie chart image for analysis"""
    if 'chart' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['chart']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the uploaded chart
        file_path = os.path.join(RESULTS_FOLDER, "chart.png")
        file.save(file_path)

        return jsonify({
            "message": "Chart uploaded successfully",
            "next_step": "Run /api/analyze-pie to analyze the chart"
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error uploading chart: {str(e)}"}), 500




def call_e2e_llama4(prompt):
    """Send prompt to E2E LLaMA 3.3 API and return the output."""
    try:
        response = client.chat.completions.create(
            model='llama_3_3_70b_instruct_fp8',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=1,
            stream=False  # Don't stream inside Streamlit
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# Generate quizzes using AI
def generate_quiz(text):
    """Generate a structured quiz from the input text using LLaMA 3.3 model."""
    prompt = f"""
    You are an educational assistant. Create a multiple-choice quiz from the text.
    Provide exactly 5 questions, each with 4 options (A, B, C, D), and mark the correct answer separately.
    Keep the language the same as the input text.

    Format the output clearly like:
    1. Question text?
       Option 1
       Option 2
       Option 3
       Option 4

    After listing all 5 questions, provide the correct answers separately in this format:

    **Correct Answers:**
    1. A
    2. C
    3. B
    4. D
    5. A

    Text:
    {text}
    """

    response_text = call_e2e_llama4(prompt)
    if not response_text:
        return {"error": "No response from LLaMA model"}

    quiz_questions = []
    correct_answers = {}

    try:
        # Split questions and correct answers
        questions_part, answers_part = response_text.split("**Correct Answers:**")

        # Process questions
        questions_blocks = questions_part.strip().split("\n\n")
        for block in questions_blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 5:
                question = lines[0].strip().replace("?", "")  # Remove ? for consistency
                options = [line.strip() for line in lines[1:5]]  # Expecting A), B), C), D)

                quiz_questions.append({
                    "question": question,
                    "options": options,
                })

        # Process correct answers
        for line in answers_part.strip().split("\n"):
            if "." in line:
                number, ans = line.strip().split(".")
                correct_answers[int(number.strip())] = ans.strip()

        # Combine into structured format
        structured_quiz = []
        for i, q in enumerate(quiz_questions):
            structured_quiz.append({
                "question": q["question"],
                "options": q["options"],
                "answer": correct_answers.get(i + 1)
            })

        return structured_quiz

    except Exception as e:
        return {"error": f"Failed to parse quiz: {str(e)}"}


# Generate structured exercises using AI
def generate_exercises(text):
    """Generate structured exercises without JSON."""
    prompt = """
    You are an educational assistant. Create structured exercises from the text.
    - 5 Fill-in-the-blank questions (missing words marked as '_____')
    - 5 Short-answer questions (1-2 sentence responses)
    - 5 Long-answer questions (detailed responses)

    Format the output clearly like:

    **Fill in the Blanks**
    1. Sentence with _____ missing.

    **Short Answer Questions**
    1. What is the importance of X?

    **Long Answer Questions**
    1. Explain how X impacts Y in detail.

    After listing all questions, provide the correct answers separately in this format:

    **Answers:**

    **Fill in the Blanks**
    1. Correct answer
    2. Correct answer

    **Short Answer Questions**
    1. Answer

    **Long Answer Questions**
    1. Answer

    Text:
    """ + text

    return call_e2e_llama4(prompt)


def extract_audio(video_path):
    """Extracts audio from a video file."""
    audio_path = video_path.replace(".mp4", ".wav")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path


def translate_text(text, target_language):
    """Translates the transcribed text into the selected language."""
    try:
        return GoogleTranslator(source="auto", target=target_language).translate(text)
    except Exception as e:
        return f"Translation Error: {str(e)}"


def detect_language(text):
    """Detects the language of the given text using langdetect."""
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails


def summarize_text(text, lang):
    """Summarizes the text and extracts key points in bullet format."""

    # Map language codes to language names for clarity in the prompt
    language_names = {
        "en": "English",
        "hi": "Hindi",
        "bn": "Bengali",
        "ta": "Tamil",
        "te": "Telugu",
        "mr": "Marathi",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam",
        "pa": "Punjabi",
        "ur": "Urdu"
    }
    try:
        lang_name = language_names.get(lang, lang)

        prompt = f"""You are a multilingual assistant.

    IMPORTANT: Your response MUST be in {lang_name} language ({lang}).
    - DO NOT respond in English. Respond ONLY in {lang_name}.
    - Extract only the most important points from the following text.
    - Use clear and concise bullet points.
    - Format your response in a structured, easy-to-read format.

    Text to summarize:
    """

        completion = client.chat.completions.create(
            model="llama_3_3_70b_instruct_fp8",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=1,
            stream=True  # Streaming enabled
        )

        summary = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                summary += chunk.choices[0].delta.content

        return summary
    except Exception as e:
        print(f"❌ API Error: {e}")
        return "Error: Failed to summarize due to API issue."


def generate_flashcards(summary_text):
    """Formats the summary into structured flashcards."""
    flashcards = []

    # Ensure summary_text is a string
    summary_text = str(summary_text)

    # Split by newline and filter out empty lines
    points = [point.strip() for point in summary_text.split("\n") if point.strip()]

    for idx, point in enumerate(points, start=1):
        flashcards.append(f"📌 **Key Point {idx}:** {point}")

    # Join the flashcards into a string and return
    return "\n".join(flashcards)


def get_latest_transcript():
    """Determine which transcript file to use."""
    if os.path.exists("translated_transcript.txt") and os.path.getsize("translated_transcript.txt") > 0:
        return "translated_transcript.txt"
    elif os.path.exists("transcript.txt") and os.path.getsize("transcript.txt") > 0:
        return "transcript.txt"
    else:
        raise FileNotFoundError("No valid transcript found. Run transcription first.")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route('/api/transcribe', methods=['POST'])
def transcribe_video():
    print("Received request...")
    print("Files:", request.files)
    print("Form data:", request.form)

    # Check if the post request has the file part
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']

    # Check if filename is empty
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get target language from form data (default to English if not provided)
    target_lang = request.form.get('target_language', 'English')

    # Validate target language
    if target_lang not in LANG_OPTIONS:
        return jsonify({"error": f"Invalid target language. Supported languages: {list(LANG_OPTIONS.keys())}"}), 400

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        video_file.save(temp_file.name)
        video_path = temp_file.name

    try:
        # Extract audio
        audio_path = extract_audio(video_path)

        # Transcribe
        transcript = model.transcribe(audio_path)["text"]

        # Translate
        translated_text = translate_text(transcript, LANG_OPTIONS[target_lang])

        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(audio_path)
        # Save original transcript
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

        # Save translated transcript
        with open("translated_transcript.txt", "w", encoding="utf-8") as f:
            f.write(translated_text)

        # Return results as JSON
        return jsonify({
            "original_transcript": transcript,
            "translated_transcript": translated_text,
            "target_language": target_lang,

        })

    except Exception as e:
        # Clean up temporary files in case of error
        if os.path.exists(video_path):
            os.unlink(video_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)

        return jsonify({"error": str(e)}), 500


@app.route('/api/summary', methods=['GET'])
def generate_summary():
    """Generate and return summary."""
    try:
        # Get the latest transcript
        transcript_file = get_latest_transcript()

        # Load text from transcript
        with open(transcript_file, "r", encoding="utf-8") as file:
            text = file.read()

        # Detect language
        detected_lang = detect_language(text)

        # Generate structured summary in bullet points
        summary_text = summarize_text(text, detected_lang)

        # Save Summary
        with open("summary.txt", "w", encoding="utf-8") as file:
            file.write(summary_text)

        return jsonify({
            "summary": summary_text,
            "language": detected_lang,
            "source_file": transcript_file
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/flashcards', methods=['GET'])
def generate_flashcards_route():
    """Generate and return flashcards."""
    try:
        # Get the latest transcript
        transcript_file = get_latest_transcript()

        # Load text from transcript
        with open(transcript_file, "r", encoding="utf-8") as file:
            text = file.read()

        # Detect language
        detected_lang = detect_language(text)

        # Generate structured summary in bullet points
        summary_text = summarize_text(text, detected_lang)

        # Generate flashcards
        flashcards = generate_flashcards(summary_text)

        # Save Flashcards
        with open("flashcards.txt", "w", encoding="utf-8") as file:
            file.write(flashcards)
        flashcard_list = [line.strip() for line in flashcards.strip().split("\n") if line.strip()]

        return jsonify({
            "flashcards": flashcard_list,
            "language": detected_lang,
            "source_file": transcript_file
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/quiz', methods=['GET'])
def quiz_route():
    transcript_text = read_transcript()

    if not transcript_text:
        return jsonify({"error": "No translated transcript file found"}), 404

    quiz = generate_quiz(transcript_text)
    return quiz, 200, {'Content-Type': 'application/json'}


import re


def parse_exercise_response(text):
    try:
        # Split main sections
        fill_blanks = re.findall(r'\*\*Fill in the Blanks\*\*\s*(.*?)\*\*Short Answer Questions\*\*', text, re.S)[
            0].strip()
        short_answers = re.findall(r'\*\*Short Answer Questions\*\*\s*(.*?)\*\*Long Answer Questions\*\*', text, re.S)[
            0].strip()
        long_answers = re.findall(r'\*\*Long Answer Questions\*\*\s*(.*?)\*\*Answers:\*\*', text, re.S)[0].strip()
        answers_section = re.findall(r'\*\*Answers:\*\*\s*(.*)', text, re.S)[0].strip()

        # Extract answers separately
        fb_answers = \
            re.findall(r'\*\*Fill in the Blanks\*\*\s*(.*?)\*\*Short Answer Questions\*\*', answers_section, re.S)[
                0].strip()
        sa_answers = \
            re.findall(r'\*\*Short Answer Questions\*\*\s*(.*?)\*\*Long Answer Questions\*\*', answers_section, re.S)[
                0].strip()
        la_answers = re.findall(r'\*\*Long Answer Questions\*\*\s*(.*)', answers_section, re.S)[0].strip()

        # Extract numbered items
        def extract_items(block):
            return [re.sub(r'^\d+\.\s*', '', line.strip()) for line in block.strip().split('\n') if line.strip()]

        return {
            "fillBlanks": extract_items(fill_blanks),
            "shortAnswer": extract_items(short_answers),
            "longAnswer": extract_items(long_answers),
            "answers": {
                "fillBlanks": extract_items(fb_answers),
                "shortAnswer": extract_items(sa_answers),
                "longAnswer": extract_items(la_answers)
            }
        }

    except Exception as e:
        print("Parsing error:", str(e))
        return None


@app.route('/api/exercise', methods=['GET'])
def exercise_route():
    transcript_text = read_transcript()

    if not transcript_text:
        return jsonify({"error": "No translated transcript file found"}), 404

    try:
        raw_text = generate_exercises(transcript_text)
        structured = parse_exercise_response(raw_text)

        if structured:
            return jsonify(structured)
        else:
            return jsonify({"error": "Failed to parse exercise response"}), 500
    except Exception as e:
        print("Exercise route error:", str(e))
        return jsonify({"error": "Internal server error"}), 500


transcription_store = {}


def get_available_languages(video_id):
    """Fetch available transcript languages for a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return [t.language_code for t in transcript_list]
    except TranscriptsDisabled:
        return []
    except Exception:
        return []


def transcribe_to_target_language(text, target_language):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Transcribe this transcript into {target_language}:\n\n{text}")
    return response.text if response else text


def generate_detailed_notes(transcript_text, language):
    prompt = f"You are a YouTube video summarizer. Summarize the transcript into key points and detailed notes in {language}."
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"{prompt}\n\n{transcript_text}")
    return response.text if response else transcript_text


def translate_notes(notes_text, target_language):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Translate the following text into {target_language}:\n\n{notes_text}")
    return response.text if response else notes_text


def extract_transcript(youtube_video_url):
    """Fetch transcript in the original language."""
    try:
        if "=" in youtube_video_url:
            video_id = youtube_video_url.split("=")[1]
        else:
            # Handle youtu.be format or other formats
            video_id = youtube_video_url.split("/")[-1]

        available_languages = get_available_languages(video_id)

        if not available_languages:
            return None, None

        transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=[available_languages[0]])
        return " ".join([i["text"] for i in transcript_text]), available_languages[0]
    except NoTranscriptFound:
        return None, None
    except Exception as e:
        return None, str(e)


@app.route('/api/transcribelink', methods=['POST'])
def transcribe_link():
    # Get data from form
    youtube_link = request.form.get('youtube_link')
    target_language = request.form.get('target_language', 'en')

    if not youtube_link:
        return jsonify({"error": "YouTube link is required"}), 400

    try:
        # Extract transcript
        transcript_text, original_language = extract_transcript(youtube_link)

        if transcript_text is None:
            return jsonify({"error": "Could not retrieve transcript for this video in any language"}), 404

        # Transcribe to target language if different
        translated_transcript = transcript_text
        if original_language != target_language:
            translated_transcript = transcribe_to_target_language(transcript_text, target_language)

        # Generate detailed notes
        detailed_notes = generate_detailed_notes(translated_transcript, target_language)

        # Generate unique ID for this transcription set
        transcription_id = str(uuid.uuid4())

        # Store the generated content
        transcription_store[transcription_id] = {
            "original_transcript": transcript_text,
            "original_language": original_language,
            "translated_transcript": translated_transcript,
            "target_language": target_language,
            "detailed_notes": detailed_notes
        }
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript_text)

        # Save translated transcript
        with open("translated_transcript.txt", "w", encoding="utf-8") as f:
            f.write(translated_transcript)

        # Return the response
        return jsonify({
            "success": True,
            "transcription_id": transcription_id,
            "original_language": original_language,
            "target_language": target_language,
            "original_transcript": transcript_text,
            "translated_transcript": translated_transcript,
            "detailed_notes": detailed_notes,
            "download_links": {
                "original_transcript": f"/download/{transcription_id}/original",
                "translated_transcript": f"/download/{transcription_id}/translated",
                "detailed_notes": f"/download/{transcription_id}/notes"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download/<transcription_id>/<file_type>', methods=['GET'])
def download_file(transcription_id, file_type):
    if transcription_id not in transcription_store:
        return jsonify({"error": "Transcription not found"}), 404

    transcription_data = transcription_store[transcription_id]

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
            if file_type == 'original':
                tmp.write(transcription_data["original_transcript"])
                filename = f"original_transcript_{transcription_data['original_language']}.txt"
            elif file_type == 'translated':
                tmp.write(transcription_data["translated_transcript"])
                filename = f"translated_transcript_{transcription_data['target_language']}.txt"
            elif file_type == 'notes':
                tmp.write(transcription_data["detailed_notes"])
                filename = f"detailed_notes_{transcription_data['target_language']}.txt"
            else:
                return jsonify({"error": "Invalid file type"}), 400

            tmp_path = tmp.name

        # Send the file
        return send_file(tmp_path, as_attachment=True, download_name=filename)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text


def extract_text_from_docx(file_path):
    """Extract text from a Word (.docx) file."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def save_text_to_file(text, filename):
    """Save text to a file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)


@app.route("/api/generate-note", methods=["POST"])
def generate_note():
    if 'file' not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files['file']
    target_language = request.form.get('target_language', 'en')
    filename = file.filename

    if not filename.lower().endswith(('.pdf', '.docx')):
        return jsonify({"error": "Unsupported file format"}), 400

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        # Extract text
        if filename.lower().endswith(".pdf"):
            notes_text = extract_text_from_pdf(temp_path)
        else:
            notes_text = extract_text_from_docx(temp_path)

        if not notes_text.strip():
            return jsonify({"error": "No text extracted from the file"}), 400

        # Translate notes
        translated_notes = translate_notes(notes_text, target_language)

        # Save to file
        save_text_to_file(translated_notes, "translated_notes.txt")

        return jsonify({
            "original_notes": notes_text,
            "translated_notes": translated_notes,
            "message": "Notes translated and saved to translated_notes.txt"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(temp_path)


class AudioTranscriber:
    def __init__(self, model_size="large-v3", device=None, compute_type="float16"):
        """Initialize speech-to-text transcriber."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"🔹 Using device: {self.device.upper()} ({'GPU' if self.device == 'cuda' else 'CPU'})")

        # Load Whisper model
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)

    def transcribe_audio(self, audio_path, input_language='en', output_language='en'):
        """
        Transcribe and optionally translate audio file.

        Args:
            audio_path (str): Path to the audio file
            input_language (str): Language of the input audio
            output_language (str): Desired output language

        Returns:
            dict: Transcription and translation results
        """
        try:
            # Transcribe audio
            segments, info = self.model.transcribe(
                audio_path, beam_size=5, language=input_language, vad_filter=True)

            # Collect transcription segments
            transcription_segments = []
            full_transcription = ""
            for segment in segments:
                text = segment.text.strip()
                transcription_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': text
                })
                full_transcription += text + " "

            # Translate if languages differ
            translated_text = full_transcription
            if output_language and output_language != input_language:
                try:
                    translator = GoogleTranslator(source=input_language, target=output_language)
                    translated_text = translator.translate(full_transcription)
                except Exception as e:
                    print(f"⚠️ Translation error: {e}")

            return {
                'input_language': input_language,
                'output_language': output_language,
                'transcription': full_transcription.strip(),
                'translated_text': translated_text.strip(),
                'segments': transcription_segments
            }

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return {'error': str(e)}


# Initialize transcriber
transcriber = AudioTranscriber()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/voice', methods=['POST'])
def transcribe_audio():
    """
    Endpoint to handle audio file transcription.

    Expects:
    - audio file in multipart/form-data
    - Optional form fields: input_language, output_language
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only MP3 and WAV allowed'}), 400

    # Get optional language parameters
    input_language = request.form.get('input_language', 'en')
    output_language = request.form.get('output_language', 'en')

    try:
        # Generate unique filename
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save file
        file.save(filepath)

        try:
            # Transcribe audio
            result = transcriber.transcribe_audio(
                filepath,
                input_language=input_language,
                output_language=output_language
            )

            # Clean up file after processing
            os.remove(filepath)

            return jsonify(result)

        except Exception as e:
            # Ensure file is deleted even if transcription fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    with app.app_context():

        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5003)

