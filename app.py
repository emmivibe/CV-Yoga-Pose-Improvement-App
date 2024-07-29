from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
import cv2
import time
import tensorflow as tf
from matplotlib import pyplot as plt
from gtts import gTTS
from playsound import playsound
import mediapipe as mp
import math
from datetime import datetime
import pyttsx3
import pythoncom
import logging
import threading
import speech_recognition as sr

app = Flask(__name__)
app.secret_key = '82db4a430db9f34c7336d1fffdbd4a8d' # Required for flashing messages

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.example.com'  # Replace with your email provider's SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@example.com'  # Replace with your email address
app.config['MAIL_PASSWORD'] = 'your_email_password'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@example.com'  # Replace with your email address

mail = Mail(app)
s = URLSafeTimedSerializer(app.secret_key)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yoga_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    accuracy = db.Column(db.Float, nullable=False)
    pose_name = db.Column(db.String(100), nullable=False)

# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    sessions = db.relationship('Session', backref='user', lazy=True)

# Pose Model
class Pose(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    right_arm = db.Column(db.Integer, nullable=False)
    left_arm = db.Column(db.Integer, nullable=False)
    right_leg = db.Column(db.Integer, nullable=False)
    left_leg = db.Column(db.Integer, nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

# Load the model
model_path = r"C:\Users\Dii\Desktop\YOGAAPP\saved_model"
model = tf.saved_model.load(model_path)
movenet = model.signatures['serving_default']
cap = cv2.VideoCapture(0)

# PoseDetector class
class PoseDetector:
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.maxHands, self.modelComplex, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

detector = PoseDetector()

# Yoga angle data
AngleData = [
    {'Name': 'tadasan', 'right_arm': 201, 'left_arm': 162, 'right_leg': 177, 'left_leg': 182},
    {'Name': 'vrksana', 'right_arm': 330, 'left_arm': 30, 'right_leg': 180, 'left_leg': 60},
    {'Name': 'balasana', 'right_arm': 200, 'left_arm': 200, 'right_leg': 20, 'left_leg': 20},
    {'Name': 'trikonasana', 'right_arm': 181, 'left_arm': 184, 'right_leg': 176, 'left_leg': 182},
    {'Name': 'virabhadrasana', 'right_arm': 170, 'left_arm': 180, 'right_leg': 160, 'left_leg': 140},
    {'Name': 'adhomukha', 'right_arm': 176, 'left_arm': 171, 'right_leg': 177, 'left_leg': 179}
]

# Add Pose data to the database if not already present
with app.app_context():
    if not Pose.query.first():
        for data in AngleData:
            pose = Pose(name=data['Name'], right_arm=data['right_arm'], left_arm=data['left_arm'], right_leg=data['right_leg'], left_leg=data['left_leg'])
            db.session.add(pose)
        db.session.commit()


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Functions to set resolution
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

# Drawing functions
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold, accuracy, threshold=80):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            if accuracy >= threshold:
                line_color = (0, 255, 0)  # Green
            else:
                line_color = (0, 0, 255)  # Red
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 2)

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold, accuracy):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold, accuracy)
        draw_keypoints(frame, person, confidence_threshold)

def compare_right_arm(right_arm, pose_index):
    pose_angles = [y for x, y in list(AngleData[pose_index].items()) if type(y) == int]
    if right_arm <= pose_angles[0]:
        acc = (right_arm / pose_angles[0]) * 100
    else:
        acc = 0

    if abs(pose_angles[0] - right_arm) <= 10:
        print("Your right arm is accurate")
    else:
        print("Your right arm is not accurate")
    return acc

def compare_left_arm(left_arm, pose_index):
    pose_angles = [y for x, y in list(AngleData[pose_index].items()) if type(y) == int]
    if left_arm <= pose_angles[1]:
        acc = (left_arm / pose_angles[1]) * 100
    else:
        acc = 0

    if abs(pose_angles[1] - left_arm) <= 10:
        print("Your left arm is accurate")
    else:
        print("Your left arm is not accurate, try again")
    return acc

def compare_right_leg(right_leg, pose_index):
    pose_angles = [y for x, y in list(AngleData[pose_index].items()) if type(y) == int]
    if right_leg <= pose_angles[2]:
        acc = (right_leg / pose_angles[2]) * 100
    else:
        acc = 0

    if abs(pose_angles[2] - right_leg) <= 10:
        print("Your right leg is accurate")
    else:
        print("Your right leg is not accurate, try again")
    return acc

def compare_left_leg(left_leg, pose_index):
    pose_angles = [y for x, y in list(AngleData[pose_index].items()) if type(y) == int]
    if left_leg <= pose_angles[3]:
        acc = (left_leg / pose_angles[3]) * 100
    else:
        acc = 0

    if abs(pose_angles[3] - left_leg) <= 10:
        print("Your left leg is accurate")
    else:
        print("Your left leg is not accurate, try again")
    return acc

def accuracyCalculation(arr):
    accArray = np.array([])
    sum = 0
    for j in range(0, len(arr) - 1, 4):
        for i in range(j, j + 4):
            sum += arr[i]
        accur = sum / 4
        accArray = np.append(accArray, accur)
    return accArray

arr = np.array([])


def play_arm_feedback():
    try:
        playsound('your_arm.mp3')  # Path to your arm feedback MP3 file
    except Exception as e:
        logging.error(f"Error in play_arm_feedback: {e}")

def play_leg_feedback():
    try:
        playsound('your_leg.mp3')  # Path to your leg feedback MP3 file
    except Exception as e:
        logging.error(f"Error in play_leg_feedback: {e}")

# Function to run arm feedback in a separate thread
def play_arm_feedback_async():
    threading.Thread(target=play_arm_feedback).start()

# Function to run leg feedback in a separate thread
def play_leg_feedback_async():
    threading.Thread(target=play_leg_feedback).start()
    
def generate_frames(selected_pose_index):
    global cap  # Use the global cap variable
    cap = cv2.VideoCapture(0)
    count = 0
    arr = []  # Initialize an empty list to store accuracy values for each body part
    timeout = 60
    timeout_start = time.time()
    feedback_count = 0  # Track the number of feedback sounds given

    while time.time() < timeout_start + timeout:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)

        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
        input_img = tf.cast(img, dtype=tf.int32)

        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

        # Initialize accuracy to 0 for coloring the connections
        accuracy = 0
        
        # Loop through people to draw connections with current accuracy
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.1, accuracy)

        frame = detector.findPose(frame, False)
        lmlist = detector.getPosition(frame, False)

        if len(lmlist) != 0:
            feedback_arms_given = False
            feedback_legs_given = False

            # Check right arm accuracy
            RightArmAngle = int(detector.findAngle(frame, 11, 13, 15))
            right_arm_accuracy = compare_right_arm(RightArmAngle, selected_pose_index)
            if count <= 16 and right_arm_accuracy != 0:
                arr.append(right_arm_accuracy)  # Append accuracy for right arm to the list
                count += 1

            # Check left arm accuracy
            LeftArmAngle = int(detector.findAngle(frame, 12, 14, 16))
            left_arm_accuracy = compare_left_arm(LeftArmAngle, selected_pose_index)
            if count <= 16 and left_arm_accuracy != 0:
                arr.append(left_arm_accuracy)  # Append accuracy for left arm to the list
                count += 1

            # Set accuracy for drawing connections
            accuracy = min(right_arm_accuracy, left_arm_accuracy)

            # Play arm feedback if both arms' accuracy is less than 80
            if right_arm_accuracy < 80 or left_arm_accuracy < 80:
                if not feedback_arms_given and feedback_count < 2:
                    play_arm_feedback_async()
                    feedback_arms_given = True
                    feedback_count += 1
            elif right_arm_accuracy >= 80 or left_arm_accuracy >= 80:
                pass  # Do nothing

            # Delay to avoid overlapping sounds
            if feedback_arms_given:
                time.sleep(3)

            # Check right leg accuracy
            RightLegAngle = int(detector.findAngle(frame, 23, 25, 27))
            right_leg_accuracy = compare_right_leg(RightLegAngle, selected_pose_index)
            if count <= 16 and right_leg_accuracy != 0:
                arr.append(right_leg_accuracy)  # Append accuracy for right leg to the list
                count += 1

            # Check left leg accuracy
            LeftLegAngle = int(detector.findAngle(frame, 24, 26, 28))
            left_leg_accuracy = compare_left_leg(LeftLegAngle, selected_pose_index)
            if count <= 16 and left_leg_accuracy != 0:
                arr.append(left_leg_accuracy)  # Append accuracy for left leg to the list
                count += 1

            # Set accuracy for drawing connections
            accuracy = min(right_leg_accuracy, left_leg_accuracy)

            # Play leg feedback if both legs' accuracy is less than 80
            if right_leg_accuracy < 80 or left_leg_accuracy < 80:
                if not feedback_legs_given and feedback_count < 2:
                    play_leg_feedback_async()
                    feedback_legs_given = True
                    feedback_count += 1
            elif right_leg_accuracy >= 80 or left_leg_accuracy >= 80:
                pass  # Do nothing

            # elif count > 16:
            #     avg_accuracy = sum(arr) / len(arr)  # Calculate average accuracy
            #     avg_accuracy_message = f"Average Accuracy: {avg_accuracy:.2f} percent"
            #     print(avg_accuracy_message)  # Print the average accuracy
            #     if avg_accuracy < 80:
            #         play_arm_feedback_async() if 'arm' in selected_pose_index else play_leg_feedback_async()


        # Update the frame with the new accuracy
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.1, accuracy)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Check for 'q' key press to x`exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        
        # Calculate and print average accuracy
    if len(arr) > 0:
        avg_accuracy = sum(arr) / len(arr)  # Calculate average accuracy
        avg_accuracy_message = f"Average Accuracy: {avg_accuracy:.2f} percent"
        print(avg_accuracy_message)  # Print the average accuracy
        if avg_accuracy < 80:
            play_arm_feedback_async() if 'arm' in selected_pose_index else play_leg_feedback_async()

    # Release the webcam and destroy all windows after the timeout
    cap.release()
    cv2.destroyAllWindows()

 # Home route
@app.route("/")
def home():
    return render_template("signin.html")

@app.route("/login", methods=["POST"])
def login():
    email = request.form["email"]
    password = request.form["password"]
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id  # Store user ID in session
        flash("Login successful!", "success")
        return redirect(url_for("menu"))
    else:
        flash("Invalid credentials. Please try again.", "danger")
        return redirect(url_for("home"))

@app.route("/register", methods=["POST"])
def register():
    username = request.form["username"]
    email = request.form["email"]
    password = generate_password_hash(request.form["password"])
    if User.query.filter_by(email=email).first():
        flash("Email already registered.", "danger")
        return redirect(url_for("home"))
    else:
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        session['user_id'] = new_user.id  # Store user ID in session
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("home"))

@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/send_reset_email', methods=['POST'])
def send_reset_email():
    email = request.form['email']
    user = User.query.filter_by(email=email).first()
    if user:
        token = s.dumps(email, salt='email-reset')
        reset_link = url_for('reset_password', token=token, _external=True)
        
        msg = Message('Password Reset Request', recipients=[email])
        msg.body = f"Hello,\n\nTo reset your password, click the following link: {reset_link}\n\nIf you did not request a password reset, please ignore this email."
        
        mail.send(msg)
        
        flash("A password reset link has been sent to your email.", "success")
    else:
        flash("Email not found. Please check and try again.", "danger")
    return redirect(url_for('home'))

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='email-reset', max_age=3600)
    except (SignatureExpired, BadTimeSignature):
        flash("The reset link is invalid or has expired.", "danger")
        return redirect(url_for('home'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        if new_password == confirm_password:
            user = User.query.filter_by(email=email).first()
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash("Your password has been reset. Please log in with your new password.", "success")
            return redirect(url_for('home'))
        else:
            flash("Passwords do not match. Please try again.", "danger")
    return render_template('reset_password.html', token=token)

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user ID from session
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))

# Menu route
@app.route("/menu")
def menu():
    return render_template("menu.html")

# Other routes
@app.route('/poses')
def poses():
    return render_template('poses.html')

@app.route('/playlist')
def playlist():
    return render_template('playlist.html')

@app.route('/lessons')
def lessons():
    return render_template('lessons.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("Please log in to view your history.", "danger")
        return redirect(url_for('home'))
    user_id = session['user_id']
    sessions = Session.query.filter_by(user_id=user_id).all()
    return render_template('history.html', sessions=sessions)

@app.route('/training')
def training():
    return render_template('training.html')

# @app.route('/start_video', methods=['POST'])
# def start_video():
#     global detection_active
#     detection_active = True
#     return jsonify({'status': 'Pose detection started'})

# @app.route('/stop_video', methods=['POST'])
# def stop_video():
#     global detection_active
#     detection_active = False
#     return jsonify({'status': 'Pose detection stopped', 'accuracy': calculate_average_accuracy(arr)})

# def calculate_average_accuracy(arr):
#     if not arr:
#         return 0
#     return sum(arr) / len(arr)

# def release_webcam():
#     cap.release()
#     cv2.destroyAllWindows()
@app.route('/stop_pose_detection', methods=['POST'])
def stop_pose_detection():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    return jsonify({'message': 'Pose detection stopped and webcam released.'})

 
@app.route('/start_pose_detection', methods=['POST'])
def start_pose_detection():
    data = request.get_json()
    pose_name = data.get('pose')
    pose = Pose.query.filter_by(name=pose_name).first()
    if pose:
        pose_index = next((index for (index, d) in enumerate(AngleData) if d["Name"] == pose_name), None)
        return jsonify({'url': url_for('video', pose_index=pose_index)})
    else:
        return jsonify({'error': 'Pose not found'}), 404

# @app.route('/stop_pose_detection', methods=['POST'])
# def stop_pose_detection():
#     cap.release()
#     cv2.destroyAllWindows()
#     return jsonify({'message': 'Pose detection stopped'})

@app.route('/video')
def video():
    pose_index = request.args.get('pose_index', default=0, type=int)
    return Response(generate_frames(pose_index), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True, port=5001)