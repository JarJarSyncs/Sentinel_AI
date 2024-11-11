import cv2
import numpy as np
from skimage import io
from batch_face import RetinaFace, LandmarkPredictor, draw_landmarks, Timer
from live_pose_estimator import SixDRep
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark
import time

def get_landmarks(frame, faces):
    ### Predict landmarks from given face co-ordinates ###
    landmarks = predictor(faces, frame, from_fd=True)
    return landmarks

def draw_landmarks_cv(frame, faces, landmarks):
    ### Draw landmarks on faces using CV2 - Possible to draw multiple faces with a For loop, however we are only interested in having one face in the frame ### 
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

def draw(live_frame, affinity_frame, holistic, mp_draw, mp_hol, hand_connections_style, custom_connections, excluded_pose_landmarks, show_advanced_face_mesh_landmarks):
    
    # Detection
    # Upload frame to GPU DOES NOT WORK DUE TO INCOMPATABILITY WITH VERSIONS D:
    #gpu_frame = cv2.cuda_GpuMat()
    #gpu_frame.upload(frame)
    #gpu_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    img_h, img_w = live_frame.shape[:2]
    #image = gpu_frame.download()  # Download image for further processing
    results = holistic.process(image)
    image.flags.writeable = True

    # Check if 'f' is pressed to toggle face landmarks, idk if we need this or not, it's just something i added, doesnt really slow down the program
    if cv2.waitKey(10) & 0xFF == ord('f'):
        show_advanced_face_mesh_landmarks = not show_advanced_face_mesh_landmarks  # Toggle the flag
        
    if show_advanced_face_mesh_landmarks and results.face_landmarks:
        mp_draw.draw_landmarks(affinity_frame, results.face_landmarks, mp_hol.FACEMESH_TESSELATION, 
                                 mp_draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_draw.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
    
     # Draw Hand Landmarks & Connections
    mp_draw.draw_landmarks(affinity_frame, results.right_hand_landmarks, mp_hol.HAND_CONNECTIONS, connection_drawing_spec=hand_connections_style)
    mp_draw.draw_landmarks(affinity_frame, results.left_hand_landmarks, mp_hol.HAND_CONNECTIONS, connection_drawing_spec=hand_connections_style)

    # Draw Upper Body Pose Landmarks & Connections
    if results.pose_landmarks:
        # Draw Upper Body Pose Landmarks
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if landmark.visibility > 0.5:
                if PoseLandmark(idx) not in excluded_pose_landmarks:
                    cv2.circle(affinity_frame, (int(landmark.x * img_w), int(landmark.y * img_h)), 5, (0, 0, 255), -1)

        # Draw Upper Body Pose Connections
        for connection in custom_connections:
            start = results.pose_landmarks.landmark[connection[0]]
            end = results.pose_landmarks.landmark[connection[1]]
            if (start.visibility > 0.5 and end.visibility > 0.5):
                if (PoseLandmark(connection[0]) not in excluded_pose_landmarks and PoseLandmark(connection[1]) not in excluded_pose_landmarks):
                    cv2.line(affinity_frame, (int(start.x * img_w), int(start.y * img_h)), (int(end.x * img_w), int(end.y * img_h)), (0, 255, 0), 2)
    return affinity_frame

def live_affinity ():
    show_advanced_face_mesh_landmarks = False #Flag to show advanced face mesh for end-user to get a more detailed face mesh, purely visual for end user.

    mp_draw = mp.solutions.drawing_utils
    mp_hol = mp.solutions.holistic

    custom_connections = list(mp_hol.POSE_CONNECTIONS)

    hand_connections_style = DrawingSpec(color=(0, 255, 0), thickness=2)

    excluded_pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32]

    holistic = mp_hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


    ### Open camera ###
    cap = cv2.VideoCapture(0)
    detector = RetinaFace(gpu_id=0) #user gpu_id=-1 for Mac to indicate CPU
    predictor = LandmarkPredictor(gpu_id=0) #user gpu_id=-1 for Mac to indicate CPU
    head_pose_estimator = SixDRep(gpu_id=0) #user gpu_id=-1 for Mac to indicate CPU
    detect_time = time.time()
    faces = None

    #while True:
    # Capture frame-by-frame
    ret, live_frame = cap.read()
    loop_time = time.time()
    
    ### NOTE: RGB values are normalized within RetinaFace ###
    ### Detect faces if none exist ###
    
    # Calculate the time difference
    elapsed_time = time.time() - detect_time

    ### Initialise a black frame ###
    black_frame = np.zeros_like(live_frame)

    # Check if n seconds has passed: The shorter the elapsed time - the more face detections are done, but also the lower the fps and efficiency
    if faces is None or elapsed_time >= 1:
        faces = detector(live_frame, cv=True, threshold=0.5)
        detect_time = time.time()
    else:
        ### This is an efficiency method of predicting the face bound-box - especially for live camera. It uses the min and max values from the results of the previous landmark 'predictor' function. Helps increase the fps rate ###
        ### However, it will not detect new faces, or when a face has gone ###
        faces = updated_bbox(landmarks)

    if len(faces) == 0:
        print("NO face is detected!")
        return ret, black_frame

    ### Predict landmarks from given face co-ordinates ###
    landmarks = predictor(faces, live_frame, from_fd=True)

    ### Estimate head pose from face ###
    pose = head_pose_estimator(faces, live_frame, input_face_type='tuple', update_dict=True)[0]
    
    ### Draw landmarks (AND/OR) pose cube ###
    black_frame = draw_landmarks(black_frame, faces[0][0], landmarks[0])
    head_pose_estimator.plot_pose_cube(black_frame, faces[0][0], **pose)

    ###
    affinity_frame = draw(live_frame, black_frame, holistic, mp_draw, mp_hol, hand_connections_style, custom_connections, excluded_pose_landmarks, show_advanced_face_mesh_landmarks)

    # Calculate and display FPS, Pitch, Yaw and Roll
    fps = 1 / (time.time() - loop_time)
    cv2.putText(black_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    #cv2.putText(black_frame, f"Pitch: {pose[0]['pitch']:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(black_frame, f"Yaw: {pose[0]['yaw']:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(black_frame, f"Roll: {pose[0]['roll']:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    
    ### Display the resulting frame ###
    #cv2.imshow('', affinity_frame)

    ### Press 'q' to exit the video window ###
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    return ret, affinity_frame
    ### Release the capture when done ###
    #cap.release()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    live_affinity()