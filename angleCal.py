import cv2
import mediapipe as mp
import numpy as np


# Initialize MediaPipe Pose module.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle < 0:
        angle += 360 
        
    return angle

# Load the image
image_path = r'C:\Users\Dii\Desktop\YOGAAPP\static\images\Tree_Pose.jpg'  # Change to your image path
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform pose detection
results = pose.process(image_rgb)

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    
    # Calculate angles for the left arm
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height]
    angle_left_arm = calculate_angle(left_shoulder, left_elbow, left_wrist)
    print('Angle of left arm:', angle_left_arm)

    # Calculate angles for the right arm
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image_width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_height]
    angle_right_arm = calculate_angle(right_shoulder, right_elbow, right_wrist)
    print('Angle of right arm:', angle_right_arm)

    # Add similar blocks for left and right legs if needed

        # Calculate angles for the left leg
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height]
    angle_left_leg = calculate_angle(left_hip, left_knee, left_ankle)
    print('Angle of left leg:', angle_left_leg)

        # Calculate angles for the right leg
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height]
    angle_right_leg = calculate_angle(right_hip, right_knee, right_ankle)
    print('Angle of right leg:', angle_right_leg)

else:
    print("No pose landmarks detected.")

# Show the image with pose landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
cv2.imshow('Yoga Pose Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
