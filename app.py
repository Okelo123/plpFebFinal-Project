# app.py - Enhanced Health Monitor Backend with Gemini API Integration
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from functools import wraps
from datetime import datetime, timedelta
import random
import json
import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import speech_recognition as sr
import requests
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import joblib
import heartpy as hp
import google.generativeai as genai

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_monitor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'models'

# Initialize Gemini
gemini_api = os.getenv("GEMINI_API_KEY")
if gemini_api:
    try:
        genai.configure(api_key=gemini_api)
        gemini = genai.GenerativeModel('gemini-pro')
        print("Gemini API initialized successfully")
    except Exception as e:
        print(f"Error initializing Gemini: {str(e)}")
        gemini = None
else:
    print("GEMINI_API_KEY not found in environment")
    gemini = None

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Create folders if not exists
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =============================================
# Login Required Decorator
# =============================================
def login_required(f):
    """Decorator to ensure user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# =============================================
# Enhanced Data Models
# =============================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    height = db.Column(db.Float)  # in cm
    weight = db.Column(db.Float)  # in kg
    role = db.Column(db.String(50), default='Patient')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    health_data = db.relationship('HealthData', backref='user', lazy=True)
    activities = db.relationship('Activity', backref='user', lazy=True)
    wearable = db.relationship('WearableDevice', backref='user', uselist=False)

class HealthData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    heart_rate = db.Column(db.Integer)
    hrv = db.Column(db.Float)  # Heart Rate Variability
    spO2 = db.Column(db.Integer)
    steps = db.Column(db.Integer)
    sleep_duration = db.Column(db.Float)  # in hours
    sleep_quality = db.Column(db.Integer)  # 1-10 scale
    stress_level = db.Column(db.Integer)   # 1-10 scale
    blood_pressure_systolic = db.Column(db.Integer)
    blood_pressure_diastolic = db.Column(db.Integer)
    temperature = db.Column(db.Float)  # body temperature
    respiration_rate = db.Column(db.Integer)  # breaths per minute
    ecg_data = db.Column(db.Text)  # JSON array of ECG values
    emotion = db.Column(db.String(50))
    health_state = db.Column(db.String(20))  # New field for health prediction

class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    activity_type = db.Column(db.String(100))
    duration = db.Column(db.Float)  # in minutes
    calories = db.Column(db.Integer)
    distance = db.Column(db.Float)  # in km
    intensity = db.Column(db.String(20))  # low, moderate, high
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class WearableDevice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    device_type = db.Column(db.String(50))  # Fitbit, Apple Watch, etc.
    device_id = db.Column(db.String(100))
    access_token = db.Column(db.String(200))
    refresh_token = db.Column(db.String(200))
    last_sync = db.Column(db.DateTime)

# Create database tables
with app.app_context():
    db.create_all()

# =============================================
# Emotion Detection Model
# =============================================
class EmotionDetector:
    def __init__(self):
        self.model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_model()
    
    def load_model(self):
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'emotion_model.h5')
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print("Emotion model loaded successfully")
            except Exception as e:
                print(f"Error loading emotion model: {str(e)}")
                self.model = None
        else:
            print("No emotion model found. Using mock responses.")
    
    def detect_emotion(self, image):
        if self.model is None:
            # Return random emotion if no model
            return random.choice(self.emotion_labels)
            
        try:
            # Convert to grayscale and detect faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                return "neutral"
            
            # Process first face found
            (x, y, w, h) = faces[0]
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Predict emotion
            predictions = self.model.predict(roi)[0]
            emotion_idx = np.argmax(predictions)
            return self.emotion_labels[emotion_idx]
        
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return "neutral"

# Initialize emotion detector
emotion_detector = EmotionDetector()

# =============================================
# Health Analysis Algorithms with Prediction Model
# =============================================
class HealthAnalyzer:
    def __init__(self):
        self.anomaly_model = self.load_anomaly_model()
        self.health_rf_model, self.health_scaler = self.load_health_model()
        self.lstm_model, self.seq_scaler = self.load_lstm_model()
    
    def load_anomaly_model(self):
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'health_anomaly_detector.joblib')
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print("Anomaly detection model loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading anomaly model: {str(e)}")
        return None
    
    def load_health_model(self):
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'health_rf_model.pkl')
        scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'health_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                print("Health prediction model loaded successfully")
                return model, scaler
            except Exception as e:
                print(f"Error loading health model: {str(e)}")
        return None, None
    
    def load_lstm_model(self):
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'health_lstm_model.h5')
        scaler_path = os.path.join(app.config['MODEL_FOLDER'], 'health_seq_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                print("LSTM health model loaded successfully")
                return model, scaler
            except Exception as e:
                print(f"Error loading LSTM model: {str(e)}")
        return None, None
    
    def predict_health_state(self, heart_rate, spO2, activity, sleep, stress):
        """Predict health state using the trained model"""
        if not self.health_rf_model or not self.health_scaler:
            return None
            
        try:
            # Create input array
            input_data = np.array([[heart_rate, spO2, activity, sleep, stress]])
            
            # Scale input
            scaled_input = self.health_scaler.transform(input_data)
            
            # Predict
            prediction = self.health_rf_model.predict(scaled_input)[0]
            probabilities = self.health_rf_model.predict_proba(scaled_input)[0]
            
            states = ['Normal', 'Warning', 'Critical']
            return {
                'prediction': states[prediction],
                'confidence': float(probabilities[prediction]),
                'probabilities': {
                    'Normal': float(probabilities[0]),
                    'Warning': float(probabilities[1]),
                    'Critical': float(probabilities[2])
                }
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def predict_health_timeseries(self, health_data):
        """Predict health state using LSTM model"""
        if not self.lstm_model or not self.seq_scaler:
            return None
            
        try:
            # Prepare input data (last 24 hours)
            if len(health_data) < 24:
                return None
                
            # Extract features in correct order
            seq_data = []
            for data in health_data[-24:]:
                seq_data.append([
                    data.heart_rate or 72,
                    data.spO2 or 98,
                    data.steps or 8000,
                    data.sleep_duration or 7.2,
                    data.stress_level or 5
                ])
                
            # Scale and reshape
            scaled_data = self.seq_scaler.transform(seq_data)
            input_data = scaled_data.reshape(1, 24, 5)
            
            # Predict
            prediction = np.argmax(self.lstm_model.predict(input_data), axis=1)[0]
            states = ['Normal', 'Warning', 'Critical']
            return states[prediction]
        except Exception as e:
            print(f"Timeseries prediction error: {str(e)}")
            return None
    
    def analyze_ecg(self, ecg_data):
        """Analyze ECG data using HeartPy"""
        try:
            # Convert JSON string to array
            if isinstance(ecg_data, str):
                ecg_data = json.loads(ecg_data)
            
            # Sample rate (typically 100-500 Hz for wearables)
            sample_rate = 250  
            
            # Process ECG data
            wd, m = hp.process(ecg_data, sample_rate)
            
            # Extract key metrics
            results = {
                'bpm': m['bpm'],
                'hrv': m['rmssd'],
                'breathing_rate': m['breathingrate'],
                'pnn50': m['pnn50'],
                'lf_hf_ratio': m['lf/hf'] if 'lf/hf' in m else 0
            }
            return results
        except Exception as e:
            print(f"ECG analysis error: {str(e)}")
            return None
    
    def detect_health_anomalies(self, health_data):
        """Detect health anomalies using Isolation Forest"""
        if self.anomaly_model is None:
            return False  # No model, return no anomaly
        
        try:
            # Prepare input features
            features = [
                health_data.heart_rate,
                health_data.spO2,
                health_data.blood_pressure_systolic,
                health_data.blood_pressure_diastolic,
                health_data.respiration_rate,
                health_data.temperature
            ]
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform([features])
            
            # Predict anomaly
            prediction = self.anomaly_model.predict(scaled_features)
            return prediction[0] == -1  # -1 indicates anomaly
        except Exception as e:
            print(f"Anomaly detection error: {str(e)}")
            return False
    
    def calculate_sleep_score(self, sleep_duration, sleep_quality):
        """Calculate comprehensive sleep score (0-100)"""
        # Weighted formula
        duration_score = min(100, sleep_duration * 12.5)  # 8 hours = 100
        quality_score = sleep_quality * 10
        return (duration_score * 0.6) + (quality_score * 0.4)
    
    def analyze_health_trends(self, user_id):
        """Analyze health trends over time"""
        # Get last 30 days of health data
        health_data = HealthData.query.filter(
            HealthData.user_id == user_id,
            HealthData.timestamp >= datetime.utcnow() - timedelta(days=30)
        ).all()
        
        if not health_data:
            return None
        
        # Convert to DataFrame for analysis
        data = [{
            'date': d.timestamp.date(),
            'heart_rate': d.heart_rate,
            'hrv': d.hrv,
            'spO2': d.spO2,
            'stress_level': d.stress_level,
            'sleep_score': self.calculate_sleep_score(d.sleep_duration, d.sleep_quality)
        } for d in health_data]
        
        df = pd.DataFrame(data)
        
        # Calculate trends
        trends = {
            'heart_rate_trend': self.calculate_trend(df, 'heart_rate'),
            'hrv_trend': self.calculate_trend(df, 'hrv'),
            'spO2_trend': self.calculate_trend(df, 'spO2'),
            'stress_trend': self.calculate_trend(df, 'stress_level'),
            'sleep_trend': self.calculate_trend(df, 'sleep_score'),
        }
        
        # Detect significant changes
        trends['critical_changes'] = self.detect_critical_changes(df)
        
        return trends
    
    def calculate_trend(self, df, column):
        """Calculate trend slope for a health metric"""
        if len(df) < 2:
            return 0
        
        # Simple linear regression for trend
        x = np.arange(len(df))
        y = df[column].values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def detect_critical_changes(self, df):
        """Detect clinically significant changes"""
        alerts = []
        
        # Heart rate change > 20 bpm sustained
        if 'heart_rate' in df.columns:
            hr_diff = df['heart_rate'].iloc[-1] - df['heart_rate'].iloc[0]
            if abs(hr_diff) > 20:
                alerts.append(f"Heart rate changed by {int(hr_diff)} bpm over 30 days")
        
        # SpO2 drop below 92%
        if 'spO2' in df.columns and df['spO2'].min() < 92:
            alerts.append("Oxygen saturation dropped below 92%")
        
        # Sleep score decline > 20%
        if 'sleep_score' in df.columns:
            sleep_diff = df['sleep_score'].iloc[-1] - df['sleep_score'].iloc[0]
            if sleep_diff < -20:
                alerts.append(f"Sleep quality declined by {int(-sleep_diff)}%")
        
        return alerts

# Initialize health analyzer
health_analyzer = HealthAnalyzer()

# =============================================
# Wearable Device Integration
# =============================================
class WearableIntegration:
    @staticmethod
    def sync_fitbit_data(user_id, access_token):
        """Sync data from Fitbit API"""
        headers = {'Authorization': f'Bearer {access_token}'}
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get heart rate data
        hr_response = requests.get(
            f'https://api.fitbit.com/1/user/-/activities/heart/date/{today}/1d.json',
            headers=headers
        )
        
        # Get sleep data
        sleep_response = requests.get(
            f'https://api.fitbit.com/1.2/user/-/sleep/date/{today}.json',
            headers=headers
        )
        
        # Get activity data
        activity_response = requests.get(
            f'https://api.fitbit.com/1/user/-/activities/date/{today}.json',
            headers=headers
        )
        
        # Process and store data
        # (Implementation depends on the actual API responses)
        # This is a simplified example
        heart_data = hr_response.json() if hr_response.status_code == 200 else None
        sleep_data = sleep_response.json() if sleep_response.status_code == 200 else None
        activity_data = activity_response.json() if activity_response.status_code == 200 else None
        
        # Create health data record
        health_data = HealthData(
            user_id=user_id,
            heart_rate=heart_data['activities-heart'][0]['value']['restingHeartRate'] if heart_data else None,
            spO2=sleep_data['summary']['oxygenSaturation']['avg'] if sleep_data else None,
            steps=activity_data['summary']['steps'] if activity_data else None,
            sleep_duration=sleep_data['summary']['totalMinutesAsleep'] / 60 if sleep_data else None,
            sleep_quality=sleep_data['summary']['efficiency'] if sleep_data else None,
            timestamp=datetime.utcnow()
        )
        
        db.session.add(health_data)
        db.session.commit()
        
        return True
    
    @staticmethod
    def sync_apple_health_data(user_id, access_token):
        """Sync data from Apple HealthKit"""
        # Apple Health integration requires special permissions
        # This is a placeholder for actual implementation
        return True

# =============================================
# Authentication Routes
# =============================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route"""
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'redirect': url_for('dashboard')
            })
        return jsonify({
            'success': False,
            'message': 'Invalid email or password'
        }), 401
    
    # For GET request, serve the login page
    return render_template('login.html')  # Your authentication HTML file

@app.route('/register', methods=['POST'])
def register():
    """User registration route"""
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name')
    
    # Check if user already exists
    if User.query.filter_by(email=email).first():
        return jsonify({
            'success': False,
            'message': 'Email already registered'
        }), 400
    
    # Hash password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    # Create new user
    new_user = User(
        username=username,
        email=email,
        password=hashed_password,
        full_name=full_name,
        role='Patient'  # Default role
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Registration successful',
        'redirect': url_for('login')
    })

@app.route('/logout')
def logout():
    """User logout route"""
    session.clear()
    return redirect(url_for('login'))

# =============================================
# Dashboard Route
# =============================================
@app.route('/')
@login_required
def dashboard():
    """Serve the main dashboard interface"""
    user = User.query.get(session['user_id'])
    return render_template('index.html', user=user)

# =============================================
# Static File Serving (CSS, JS, etc.)
# =============================================
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# =============================================
# Enhanced Health Data Routes
# =============================================
@app.route('/store_health_data', methods=['POST'])
@login_required
def store_health_data():
    data = request.get_json()
    
    # Create new health data record
    new_health_data = HealthData(
        user_id=session['user_id'],
        heart_rate=data.get('heart_rate'),
        hrv=data.get('hrv'),
        spO2=data.get('spO2'),
        steps=data.get('steps'),
        sleep_duration=data.get('sleep_duration'),
        sleep_quality=data.get('sleep_quality'),
        stress_level=data.get('stress_level'),
        blood_pressure_systolic=data.get('systolic'),
        blood_pressure_diastolic=data.get('diastolic'),
        temperature=data.get('temperature'),
        respiration_rate=data.get('respiration_rate'),
        ecg_data=json.dumps(data.get('ecg_data')) if data.get('ecg_data') else None,
        emotion=data.get('emotion')
    )
    
    db.session.add(new_health_data)
    db.session.commit()
    
    # Perform health analysis
    anomaly_detected = health_analyzer.detect_health_anomalies(new_health_data)
    
    # Analyze ECG if available
    ecg_analysis = None
    if data.get('ecg_data'):
        ecg_analysis = health_analyzer.analyze_ecg(data.get('ecg_data'))
    
    # Predict health state
    health_prediction = health_analyzer.predict_health_state(
        heart_rate=data.get('heart_rate') or 72,
        spO2=data.get('spO2') or 98,
        activity=data.get('steps') or 8000,
        sleep=data.get('sleep_duration') or 7.2,
        stress=data.get('stress_level') or 5
    )
    
    # Store prediction in database
    if health_prediction:
        new_health_data.health_state = health_prediction['prediction']
        db.session.commit()
    
    # Add time-series prediction if enough data exists
    health_data = HealthData.query.filter_by(user_id=session['user_id']).order_by(HealthData.timestamp.desc()).limit(24).all()
    if len(health_data) >= 24:
        timeseries_pred = health_analyzer.predict_health_timeseries(health_data)
        if timeseries_pred:
            health_prediction['timeseries_prediction'] = timeseries_pred
    
    return jsonify({
        'success': True,
        'anomaly_detected': anomaly_detected,
        'ecg_analysis': ecg_analysis,
        'health_prediction': health_prediction
    })

@app.route('/get_health_trends')
@login_required
def get_health_trends():
    trends = health_analyzer.analyze_health_trends(session['user_id'])
    return jsonify(trends)

# =============================================
# Enhanced Webcam Processing
# =============================================
@app.route('/process_webcam', methods=['POST'])
@login_required
def process_webcam():
    # Get image data from frontend
    data = request.get_json()
    image_data = data['image'].split(',')[1]  # Remove the data URL prefix
    
    # Convert base64 to image
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect emotion
    emotion = emotion_detector.detect_emotion(img)
    
    # Generate health metrics (in real app, this would use physiological signals)
    heart_rate = random.randint(60, 100)
    spO2 = random.randint(95, 100)
    stress_level = random.randint(1, 10)
    
    # Calculate stress level based on emotion
    emotion_stress_map = {
        'angry': 9, 'disgust': 7, 'fear': 8, 
        'happy': 2, 'sad': 6, 'surprise': 5, 'neutral': 4
    }
    stress_level = emotion_stress_map.get(emotion, 5)
    
    # Predict health state
    health_prediction = health_analyzer.predict_health_state(
        heart_rate=heart_rate,
        spO2=spO2,
        activity=0,
        sleep=7.2,
        stress=stress_level
    )
    
    # Store in database
    health_data = HealthData(
        user_id=session['user_id'],
        emotion=emotion,
        stress_level=stress_level,
        health_state=health_prediction['prediction'] if health_prediction else None,
        timestamp=datetime.utcnow()
    )
    db.session.add(health_data)
    db.session.commit()
    
    # Generate recommendations
    recommendations = []
    if health_prediction and health_prediction['prediction'] != 'Normal':
        recommendations.append(f"Health state: {health_prediction['prediction']} - Consult your doctor")
    elif emotion in ['angry', 'fear', 'sad']:
        recommendations.append("You seem stressed. Try deep breathing exercises.")
    elif heart_rate > 85:
        recommendations.append("Your heart rate is elevated. Consider relaxing activities.")
    else:
        recommendations.append("Your vitals look good! Maintain your healthy habits.")
    
    return jsonify({
        'emotion': emotion,
        'heart_rate': heart_rate,
        'spO2': spO2,
        'stress_level': stress_level,
        'health_prediction': health_prediction,
        'recommendations': recommendations
    })

# =============================================
# Wearable Device Integration Routes
# =============================================
@app.route('/connect_wearable', methods=['POST'])
@login_required
def connect_wearable():
    data = request.get_json()
    device_type = data.get('device_type')
    access_token = data.get('access_token')
    refresh_token = data.get('refresh_token')
    device_id = data.get('device_id')
    
    # Check if device already connected
    wearable = WearableDevice.query.filter_by(user_id=session['user_id']).first()
    
    if wearable:
        # Update existing device
        wearable.access_token = access_token
        wearable.refresh_token = refresh_token
        wearable.device_id = device_id
    else:
        # Create new device connection
        wearable = WearableDevice(
            user_id=session['user_id'],
            device_type=device_type,
            access_token=access_token,
            refresh_token=refresh_token,
            device_id=device_id,
            last_sync=datetime.utcnow()
        )
        db.session.add(wearable)
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/sync_wearable_data', methods=['POST'])
@login_required
def sync_wearable_data():
    wearable = WearableDevice.query.filter_by(user_id=session['user_id']).first()
    if not wearable:
        return jsonify({'error': 'No wearable device connected'}), 400
    
    # Sync based on device type
    if wearable.device_type.lower() == 'fitbit':
        success = WearableIntegration.sync_fitbit_data(session['user_id'], wearable.access_token)
    elif wearable.device_type.lower() == 'apple':
        success = WearableIntegration.sync_apple_health_data(session['user_id'], wearable.access_token)
    else:
        return jsonify({'error': 'Unsupported device type'}), 400
    
    if success:
        wearable.last_sync = datetime.utcnow()
        db.session.commit()
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to sync data'}), 500

# =============================================
# Enhanced Chatbot with Gemini API Integration
# =============================================
@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot():
    data = request.get_json()
    message = data.get('message')
    
    if not message or not message.strip():
        return jsonify({'reply': "Please provide a message to chat."})
    
    # Get user health data for context
    user = User.query.get(session['user_id'])
    
    # Initialize health context
    health_context = "Patient profile: "
    
    if user:
        # Basic user info
        if user.age: health_context += f"{user.age} year old "
        if user.gender: health_context += f"{user.gender}, "
        if user.height: health_context += f"{user.height}cm, "
        if user.weight: health_context += f"{user.weight}kg. "
        
        # Get latest health data
        health_data = HealthData.query.filter_by(user_id=user.id).order_by(HealthData.timestamp.desc()).first()
        if health_data:
            health_context += "\n\nLatest health metrics:"
            if health_data.heart_rate is not None: 
                health_context += f"\n- Heart rate: {health_data.heart_rate} bpm"
            if health_data.spO2 is not None: 
                health_context += f"\n- SpO2: {health_data.spO2}%"
            if health_data.stress_level is not None: 
                health_context += f"\n- Stress level: {health_data.stress_level}/10"
            if health_data.emotion: 
                health_context += f"\n- Last emotion: {health_data.emotion}"
            if health_data.health_state: 
                health_context += f"\n- Health state: {health_data.health_state}"
    
    # If we don't have any context, use a fallback
    if health_context == "Patient profile: ":
        health_context = "No health data available. The user is asking a general health question."
    
    # Create prompt for Gemini
    prompt = f"""
    You are HealthBot, an AI health assistant. 
    Provide helpful, professional advice based on the user's health data.
    
    {health_context}
    
    User question: {message}
    
    Response guidelines:
    - Be concise and medically accurate
    - Focus on actionable advice
    - Include specific recommendations when appropriate
    - Use simple language avoiding medical jargon
    - Be empathetic and supportive
    - If the question is not health-related, politely explain you can only answer health questions
    """
    
    print(f"Chatbot prompt:\n{prompt}")
    
    if not gemini:
        return jsonify({'reply': "Health assistant is currently unavailable. Please try again later."})
    
    try:
        # Call Gemini API
        response = gemini.generate_content(prompt)
        
        # Ensure we have valid text response
        if response.text and response.text.strip():
            reply = response.text.strip()
            return jsonify({'reply': reply})
        else:
            print("Gemini returned empty response")
            return jsonify({'reply': "I didn't get a response. Please try again."})
    
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        # Return a helpful message instead of an error
        return jsonify({'reply': "I'm having trouble connecting to the health assistant service. Please try again later."})

# =============================================
# Speech Recognition Route
# =============================================
@app.route('/speech_to_text', methods=['POST'])
@login_required
def speech_to_text():
    # Get audio file from request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save the audio file temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    audio_file.save(temp_path)
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return jsonify({'text': text})
    
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Speech recognition service error: {str(e)}'}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# =============================================
# Run the app
# =============================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)