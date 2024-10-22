import cv2
import time
import numpy as np
#import requests
#import base64
import mediapipe as mp
import numpy as np
from batch_face import RetinaFace, LandmarkPredictor, draw_landmarks, Timer
from live_pose_estimator import SixDRep
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import torch
print(torch.cuda.is_available())

from face_eye_crop import get_input_data, draw_axis

def estimate_gaze(landmarks, frame_width, frame_height):
    left_eye_center = np.mean([landmarks[i] for i in LEFT_IRIS], axis=0)
    right_eye_center = np.mean([landmarks[i] for i in RIGHT_IRIS], axis=0)

    # Gaze direction vector (difference between right and left eye center positions)
    gaze_direction = (right_eye_center[0] - left_eye_center[0], right_eye_center[1] - left_eye_center[1])
    return gaze_direction

# Define the face detection and head pose functions
def get_landmarks(frame, faces):
    landmarks = predictor(faces, frame, from_fd=True)
    return landmarks

def draw_landmarks_cv(frame, faces, landmarks):
    frame = draw_landmarks(frame, faces[0][0], landmarks[0])
    return frame

def get_head_pose(frame, faces_pose):
    head_poses = head_pose_estimator(faces_pose, frame, input_face_type='tuple', update_dict=True)
    return head_poses

def draw_head_pose_cube_cv(frame, faces, pose):
    head_pose_estimator.plot_pose_cube(frame, faces[0][0], **pose)

def updated_bbox(landmarks):
    ldm_new = landmarks[0]
    (x1, y1), (x2, y2) = ldm_new.min(0), ldm_new.max(0)
    box_new = np.array([x1, y1, x2, y2])
    box_new[:2] -= 10
    box_new[2:] += 10
    faces = [[box_new, None, None]]
    return faces

# Initialize MediaPipe for pose, face, and hand detection
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hol = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE = [7, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163]
LEFT_EYE_CONNECTIONS = [(LEFT_EYE[i], LEFT_EYE[i + 1]) for i in range(len(LEFT_EYE) - 1)]

RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
RIGHT_EYE_CONNECTIONS = [(RIGHT_EYE[i], RIGHT_EYE[i + 1]) for i in range(len(RIGHT_EYE) - 1)]

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE = [0, 4, 6]
NOSE_CONNECTIONS = [(NOSE[i], NOSE[i + 1]) for i in range(len(NOSE) - 1)]
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
MOUTH_OUTER_CONNECTIONS = [(MOUTH_OUTER[i], MOUTH_OUTER[i + 1]) for i in range(len(MOUTH_OUTER) - 1)]
FACE_OVAL_CONNECTIONS = [(FACE_OVAL[i], FACE_OVAL[i + 1]) for i in range(len(FACE_OVAL) - 1)]
FACE_OVAL_CONNECTIONS.append((FACE_OVAL[-1], FACE_OVAL[0]))
LEFT_EYE_CONNECTIONS.append((LEFT_EYE[-1], LEFT_EYE[0]))
RIGHT_EYE_CONNECTIONS.append((RIGHT_EYE[-1], RIGHT_EYE[0]))

BUFFER_PERIOD = 100  # Number of frames to ignore before starting detection
pose_estimation_state = False  # True when target detected
target_frames_counter = 0
# Add a scale factor for the iris circles
scale_factor = 0.5

custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
custom_connections = list(mp_hol.POSE_CONNECTIONS)
custom_face_landmark_style = DrawingSpec(color=(0, 0, 255), thickness=1)
hand_connections_style = DrawingSpec(color=(0, 255, 0), thickness=2)
excluded_pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32]

holistic = mp_hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

for landmark in excluded_pose_landmarks:
    # Change the way the excluded landmarks are drawn
    custom_style[landmark] = DrawingSpec(color=(255, 255, 0), thickness=None)
    # Remove all connections which contain these landmarks
    custom_connections = [
        connection_tuple
        for connection_tuple in custom_connections
        if landmark not in connection_tuple
    ]

# Open camera
cap = cv2.VideoCapture(r"C:\Users\scout\Desktop\Sentinel_AI-main\live_detection\IMG_5307.mp4")


# Initialize detectors and pose estimator
detector = RetinaFace(gpu_id=0)  # Use CPU
predictor = LandmarkPredictor(gpu_id=0)  # Use CPU
head_pose_estimator = SixDRep(gpu_id=0)  # Use CPU
holistic = mp_hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

detect_time = time.time()
faces = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Create a black canvas (same size as frame)
    black_canvas = np.zeros_like(frame)

    loop_time = time.time()

    input_data = get_input_data(frame)  

    if input_data is None or len(input_data) == 0:
            continue
    
    for face in input_data:
            box = face['box']

            # Get face bounding box coordinates
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            # Adjust the bounding box slightly
            x_min = max(0, x_min - int(0.2 * bbox_height))
            y_min = max(0, y_min - int(0.2 * bbox_width))
            x_max += int(0.2 * bbox_height)
            y_max += int(0.2 * bbox_width)

            # Get head pose angles
            hp = face['p_pred_deg']
            hy = face['y_pred_deg']
            hr = face['r_pred_deg']

            # Draw head pose axes on the black canvas
            draw_axis(black_canvas, hy, hp, hr, x_min + int(.5 * (x_max - x_min)), y_min + int(.5 * (y_max - y_min)), size=130)

    #if face_data:
     #   for face_info in face_data:
      #      right_eye_img = face_info['image']
      #      cv2.imshow('Right and Left Eye', right_eye_img)

          # Get head pose angles
  #          yaw, pitch, roll = face_info['y_pred_deg'], face_info['p_pred_deg'], face_info['r_pred_deg']
   #         box = face_info['box']

            # Draw the eye axis using the imported draw_eye_axis function
     #       black_canvas = draw_eye_axis(black_canvas, yaw, pitch, roll, int(box[0]), int(box[1]))

    # Calculate the time difference for face detection efficiency
    elapsed_time = time.time() - detect_time

    # Detect faces every second or if no face exists
    if faces is None or elapsed_time >= 1:
        faces = detector(frame, cv=True, threshold=0.5)
        detect_time = time.time()
    else:
        faces = updated_bbox(landmarks)

    if len(faces) == 0:
        print("No face detected!")
        continue

    # Predict landmarks
    landmarks = get_landmarks(frame, faces)

    # Estimate head pose
    pose = get_head_pose(frame, faces)

    # Process holistic pose and face detection
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = holistic.process(image)
    face_results = face_mesh.process(image)

    # Convert back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # Draw Face, Nose, Mouth connections
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Get the mesh points
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in face_landmarks.landmark
            ])
            # Calculate gaze direction using the eye landmarks
            gaze_direction = estimate_gaze(mesh_points, img_w, img_h)

            # Draw iris landmarks for visualization
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Draw circles for left and right iris
            cv2.circle(black_canvas, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(black_canvas, center_right, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA)

            # Display the gaze direction on the screen
            cv2.putText(black_canvas, f"Gaze Direction: {gaze_direction}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Draw face landmarks and connections
            for connection in FACE_OVAL_CONNECTIONS:
                start = face_landmarks.landmark[connection[0]]
                end = face_landmarks.landmark[connection[1]]
                cv2.line(black_canvas, (int(start.x * img_w), int(start.y * img_h)), (int(end.x * img_w), int(end.y * img_h)), (0, 255, 0), 2)
            
            # Draw nose connections
            for connection in NOSE_CONNECTIONS:
                start = face_landmarks.landmark[connection[0]]
                end = face_landmarks.landmark[connection[1]]
                cv2.line(black_canvas, (int(start.x * img_w), int(start.y * img_h)), (int(end.x * img_w), int(end.y * img_h)), (0, 255, 0), 2)
                
            # Draw mouth connections
            for connection in MOUTH_OUTER_CONNECTIONS:
                start = face_landmarks.landmark[connection[0]]
                end = face_landmarks.landmark[connection[1]]
                cv2.line(black_canvas, (int(start.x * img_w), int(start.y * img_h)), (int(end.x * img_w), int(end.y * img_h)), (0, 255, 0), 2)

    # Draw hand landmarks and connections
    mp_draw.draw_landmarks(black_canvas, results.right_hand_landmarks, mp_hol.HAND_CONNECTIONS, connection_drawing_spec=hand_connections_style)
    mp_draw.draw_landmarks(black_canvas, results.left_hand_landmarks, mp_hol.HAND_CONNECTIONS, connection_drawing_spec=hand_connections_style)

    # Draw Upper Body Pose Landmarks & Connections
    if results.pose_landmarks:
        # Draw Upper Body Pose Landmarks
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if landmark.visibility > 0.5:
                if PoseLandmark(idx) not in excluded_pose_landmarks:
                    cv2.circle(black_canvas, (int(landmark.x * img_w), int(landmark.y * img_h)), 5, (0, 0, 255), -1)

        # Draw Upper Body Pose Connections
        for connection in custom_connections:
            start = results.pose_landmarks.landmark[connection[0]]
            end = results.pose_landmarks.landmark[connection[1]]
            if (start.visibility > 0.5 and end.visibility > 0.5):
                if (PoseLandmark(connection[0]) not in excluded_pose_landmarks and PoseLandmark(connection[1]) not in excluded_pose_landmarks):
                    cv2.line(black_canvas, (int(start.x * img_w), int(start.y * img_h)), (int(end.x * img_w), int(end.y * img_h)), (0, 255, 0), 2)

    # Draw irises
    if face_results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in face_results.multi_face_landmarks[0].landmark
            ])

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv2.circle(black_canvas, center_left, int(l_radius * scale_factor), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(black_canvas, center_right, int(r_radius * scale_factor), (0, 255, 0), 1, cv2.LINE_AA)

    # Draw landmarks and pose cube on the black canvas
    black_canvas = draw_landmarks_cv(black_canvas, faces, landmarks)
    draw_head_pose_cube_cv(black_canvas, faces, pose[0])

    # Calculate and display FPS, pitch, yaw, and roll
    fps = 1 / (time.time() - loop_time)
    cv2.putText(black_canvas, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(black_canvas, f"Pitch: {pose[0]['pitch']:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(black_canvas, f"Yaw: {pose[0]['yaw']:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(black_canvas, f"Roll: {pose[0]['roll']:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show results
    cv2.imshow('Part Affinity Display', black_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
