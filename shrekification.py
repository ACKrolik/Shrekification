import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize mediapipe face detection and face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Load Shrek face images with alpha channel (transparency)
shrek_happy = cv2.imread('shrek_happy.png', cv2.IMREAD_UNCHANGED)
shrek_sad = cv2.imread('shrek_sad.png', cv2.IMREAD_UNCHANGED)
shrek_surprised = cv2.imread('shrek_surprised.png', cv2.IMREAD_UNCHANGED)

# OpenCV camera setup
cap = cv2.VideoCapture(0)

# Initialize mediapipe
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)

# Initialize the toggle state for overlaying Shrek's face
toggle_shrek_overlay = False

# Initialize FPS calculation
prev_time = 0

def detect_emotion(face_landmarks):
    left_mouth = np.array([face_landmarks[61].x, face_landmarks[61].y])
    right_mouth = np.array([face_landmarks[291].x, face_landmarks[291].y])
    top_lip = np.array([face_landmarks[13].x, face_landmarks[13].y])
    bottom_lip = np.array([face_landmarks[14].x, face_landmarks[14].y])

    mouth_width = np.linalg.norm(left_mouth - right_mouth)
    mouth_openness = np.linalg.norm(top_lip - bottom_lip)
    
    left_eye_top = np.array([face_landmarks[159].x, face_landmarks[159].y])
    left_eye_bottom = np.array([face_landmarks[145].x, face_landmarks[145].y])
    eye_openness = np.linalg.norm(left_eye_top - left_eye_bottom)

    # Stricter sad threshold: smaller mouth width and very little mouth openness
    sad_threshold = mouth_openness / mouth_width < 0.15 and mouth_width < 0.45
    
    # Adjust surprised threshold: rounder mouth (more vertical openness) I can't get this one to work :(
    surprised_threshold = (mouth_openness / mouth_width > 0.7 and
                           mouth_width > 0.3 and
                           eye_openness > 0.05)

    # Happy threshold remains the same
    happy_threshold = mouth_openness / mouth_width > 0.35 and mouth_width > 0.5

    if happy_threshold:
        return 'happy'
    elif sad_threshold:
        return 'sad'
    elif surprised_threshold:
        return 'surprised'

    return 'neutral'


def overlay_shrek_face(frame, face_landmarks, emotion):
    left_eye = np.array([face_landmarks[33].x * frame.shape[1], face_landmarks[33].y * frame.shape[0]])
    right_eye = np.array([face_landmarks[263].x * frame.shape[1], face_landmarks[263].y * frame.shape[0]])

    if emotion == 'happy':
        shrek_face = shrek_happy
        scaling_factor = 3.5
    elif emotion == 'sad':
        shrek_face = shrek_sad
        scaling_factor = 4.0
    elif emotion == 'surprised':
        shrek_face = shrek_surprised
        scaling_factor = 1.8
    else:
        shrek_face = shrek_happy
        scaling_factor = 3.5

    eye_distance = np.linalg.norm(left_eye - right_eye)
    width = int(eye_distance * scaling_factor)
    height = int(width * shrek_face.shape[0] / shrek_face.shape[1])

    shrek_resized = cv2.resize(shrek_face, (width, height), interpolation=cv2.INTER_AREA)
    alpha_mask = shrek_resized[:, :, 3] / 255.0
    inverse_alpha = 1.0 - alpha_mask

    x_offset = int(left_eye[0] - width // 2)
    y_offset = int(left_eye[1] - height // 2)

    x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + width)
    y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + height)

    shrek_cropped = shrek_resized[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]
    alpha_mask_cropped = alpha_mask[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]
    inverse_alpha_cropped = inverse_alpha[y1 - y_offset:y2 - y_offset, x1 - x_offset:x2 - x_offset]

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (alpha_mask_cropped * shrek_cropped[:, :, c] +
                                  inverse_alpha_cropped * frame[y1:y2, x1:x2, c])

    return frame

def apply_blurred_background(frame, face_landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = np.array([[int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])] 
                       for landmark in face_landmarks])
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    blurred_frame = cv2.GaussianBlur(frame, (35, 35), 0)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    blurred_background = cv2.bitwise_and(blurred_frame, blurred_frame, mask=cv2.bitwise_not(mask))
    output = cv2.add(frame, blurred_background)
    return output

def display_histogram(frame, hist_plots):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        hist_plots[i].set_ydata(hist)
        ax.set_ylim([0, max(np.max(hist), 5000)])

    plt.draw()
    plt.pause(0.001)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def draw_instructions(frame):
    cv2.putText(frame, "'Space' to toggle Shrek overlay", (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# Initialize the histogram plot
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
color = ('b', 'g', 'r')
hist_plots = []
for col in color:
    hist_plot, = ax.plot(np.zeros(256), color=col)
    hist_plots.append(hist_plot)

ax.set_xlim([0, 256])
ax.set_ylim([0, 3000])
ax.set_title('Histogram for color scale picture')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_detection_results = face_detection.process(rgb_frame)

    if face_detection_results.detections:
        face_mesh_results = face_mesh.process(rgb_frame)
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                emotion = detect_emotion(face_landmarks.landmark)

                frame = apply_blurred_background(frame, face_landmarks.landmark)

                if toggle_shrek_overlay:
                    if face_landmarks.landmark:
                        frame = overlay_shrek_face(frame, face_landmarks.landmark, emotion)
                else:
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    draw_fps(frame, fps)
    draw_instructions(frame)

    display_histogram(frame, hist_plots)

    cv2.imshow('Shrek Deepfake with UI', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        toggle_shrek_overlay = not toggle_shrek_overlay
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
