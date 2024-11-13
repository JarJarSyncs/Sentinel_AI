import cv2
import numpy as np
from skimage import io
from batch_face import RetinaFace, LandmarkPredictor, draw_landmarks, Timer
from live_pose_estimator import SixDRep
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark
import time
from face_eye_crop import get_input_data, draw_eye_gaze
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from sixdrepnet import utils
from sixdrepnet.model import SixDRepNet

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

class SixthEyeNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(9, 26, 3)
        self.fc1 = nn.Linear(3432, 600)
        self.fc2 = nn.Linear(600, 50)
        self.fc3 = nn.Linear(53, 2)

    def forward(self, x):
        x, head_pos = x
        head_pos = head_pos
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, head_pos), 1)
        x = self.fc3(x)
        return x

class LiveAffinity:
    def __init__(self):
        # Flag to show advanced face mesh for end-user to get a more detailed face mesh, purely visual for end user.
        self.show_advanced_face_mesh_landmarks = True

        # MediaPipe drawing utilities and holistic model for facial, hand, and pose detections
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hol = mp.solutions.holistic

        # Custom connections for pose landmarks
        self.custom_connections = list(self.mp_hol.POSE_CONNECTIONS)

        # Define drawing style for hand landmarks
        self.hand_connections_style = DrawingSpec(color=(0, 255, 0), thickness=2)

        # List of excluded pose landmarks for specific use cases
        self.excluded_pose_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32]

        # Initialize models and detectors
        self.holistic = self.mp_hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.detector = RetinaFace(gpu_id=0)  # Use gpu_id=-1 for CPU on Mac
        self.predictor = LandmarkPredictor(gpu_id=0)  # Use gpu_id=-1 for CPU on Mac
        self.head_pose_estimator = SixDRep(gpu_id=0)  # Use gpu_id=-1 for CPU on Mac
        self.detect_time = time.time()  # Initialize detection time for tracking intervals
        self.faces = None  # Initialize face detections

        self.device = torch.device('cuda') #-1 for mac
        self.net = SixthEyeNet()
        self.EYE_MODEL_PATH = 'D:/Uni/git/Sentinel_AI/eye-gaze-data-loader/models/sixth_eye_net_combined.pth'
        self.bw = False
        self.net.load_state_dict(torch.load(self.EYE_MODEL_PATH))
        self.net.to(self.device)

        self.transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((70, 210)),
                                     transforms.ToTensor()])
        
        self.model = SixDRepNet(backbone_name='RepVGG-B1g2',backbone_file='',deploy=True,pretrained=False)
        self.model.to(self.device)

    def live_affinity(self, ret, live_frame):
        loop_time = time.time()  # Track start time of the loop
        
        # Calculate the time difference since last detection
        elapsed_time = time.time() - self.detect_time

        # Initialize a black frame for overlaying landmarks
        black_frame = np.zeros_like(live_frame)
        # Check if 'faces' is None or if a second has passed to re-detect faces
        #if self.faces is None or elapsed_time >= 1:
            # Detect faces with RetinaFace model
        #    self.faces = self.detector(live_frame, cv=True, threshold=0.5)
        #    self.detect_time = time.time()  # Reset detection time after each detection
        #else:
            # Update bounding box based on previous landmark positions for efficiency
        #    self.faces = updated_bbox(self.landmarks)

        #if len(self.faces) == 0:
        #    print("NO face is detected!")
        #    return ret, black_frame
        # Predict landmarks based on detected face coordinates
        #self.landmarks = self.predictor(self.faces, live_frame, from_fd=True)

        # Estimate head pose from the detected face
        #pose = self.head_pose_estimator(self.faces, live_frame, input_face_type='tuple', update_dict=True)[0]

        # Draw landmarks and pose cube on the black frame
        #black_frame = draw_landmarks(black_frame, self.faces[0][0], self.landmarks[0])
        #self.head_pose_estimator.plot_pose_cube(black_frame, self.faces[0][0], **pose)

        # Check if 'f' is pressed to toggle face landmarks, idk if we need this or not, it's just something i added, doesnt really slow down the program
        if cv2.waitKey(10) & 0xFF == ord('f'):
            self.show_advanced_face_mesh_landmarks = not self.show_advanced_face_mesh_landmarks  # Toggle the flag
        # Draw the final frame with the landmarks and pose estimation
        affinity_frame = draw(
            live_frame, black_frame, self.holistic, self.mp_draw, self.mp_hol,
            self.hand_connections_style, self.custom_connections, 
            self.excluded_pose_landmarks, self.show_advanced_face_mesh_landmarks
        )
        
        input_data = get_input_data(live_frame, self.transformations, self.device, self.model, self.detector)

        if input_data is not None:
            if len(input_data) != 0:
                for face in input_data:
                    draw_eye_gaze(face, self.net, self.device, self.transforms, affinity_frame)
        else:
            cv2.putText(affinity_frame, f"No Face Detected!", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Calculate and display FPS on the black frame
        fps = 1 / (time.time() - loop_time)
        cv2.putText(affinity_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return ret, affinity_frame
        ### Release the capture when done ###
        #cap.release()
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    live_affinity_instance = LiveAffinity()
    cap = cv2.VideoCapture(0)
    while True:
        # Capture a frame from the video feed
        ret, live_frame = cap.read()
        # Process the frame using live_affinity to get landmarks and pose
        if ret:
            ret, processed_frame = live_affinity_instance.live_affinity(ret, live_frame)
            cv2.imshow('Part Affinity Display', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # Release the capture when done
    cap.release()