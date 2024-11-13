from batch_face import (
    RetinaFace,
    SixDRep
)
from sixdrepnet.model import SixDRepNet
import os
import numpy as np
import cv2
from math import cos, sin

import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms
from PIL import Image
from sixdrepnet import utils

# image transformations
transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

detector = RetinaFace(gpu_id=-1) # MacOS no cuda
cam = 1
device = torch.device('cpu')
model = SixDRepNet(backbone_name='RepVGG-B1g2',
                   backbone_file='',
                   deploy=True,
                   pretrained=False)

def get_input_data(image, transformations, device, model, detector, offset_coeff=1) -> dict:
    coeff = 1280 / image.shape[1]
    resized_image = cv2.resize(image, (1280, int(image.shape[0]*coeff)))
    try:
        with torch.no_grad():
            faces = detector(resized_image)
            if len(faces) == 0:
                #Sprint("NO face is detected!")
                return None
            result = []
            for box, landmarks, score in faces:

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])

                x_min2 = int(box[0])
                y_min2 = int(box[1])
                x_max2 = int(box[2])
                y_max2 = int(box[3])

                x_3 = int(landmarks[0][0])
                y_3 = int(landmarks[0][1])
                x_4 = int(landmarks[1][0])
                y_4 = int(landmarks[1][1])

                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max += int(0.2*bbox_height)
                y_max += int(0.2*bbox_width)

                img = resized_image[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).to(device)

                R_pred = model(img)

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi

                curr = {'p_pred_deg': euler[:, 0],
                        'y_pred_deg': euler[:, 1],
                        'r_pred_deg': euler[:, 2]
                        }

                offset = abs(((x_3 - x_min2)/2 + (x_max2-x_4)/2)/2)
                x_offset = int(offset*1.2*offset_coeff)
                y_offset = int(offset*0.8*offset_coeff)

                y_3_min = int((y_3 - y_offset) / coeff)
                y_3_max = int((y_3 + y_offset) / coeff)
                x_3_min = int((x_3 - x_offset) / coeff)
                x_3_max = int((x_3 + x_offset) / coeff)

                y_4_min = int((y_4 - y_offset) / coeff)
                y_4_max = int((y_4 + y_offset) / coeff)
                x_4_min = int((x_4 - x_offset) / coeff)
                x_4_max = int((x_4 + x_offset) / coeff)

                right_eye = image[y_3_min:y_3_max, x_3_min: x_3_max]
                left_eye = image[y_4_min:y_4_max, x_4_min: x_4_max]
                left_eye = cv2.resize(
                    left_eye, (right_eye.shape[1], right_eye.shape[0]))
                curr['image'] = cv2.hconcat([right_eye, left_eye])
                curr['box'] = list(map(lambda x: x/coeff, box))
                curr['landmarks'] = list(
                    map(lambda y: list(map(lambda x: x/coeff, y)), landmarks))
                result.append(curr)
            return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def draw_eye_axis(img, yaw, pitch, roll, tdx, tdy, size=100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    x = size * (sin(yaw)) + tdx
    y = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x), int(y)), (255, 255, 0), 3)

    return img

def draw_eye_gaze(face, net, device, transforms, affinity_frame):
    box = face['box']

    # Print the location of each face in this image
    x_min = int(box[0])
    y_min = int(box[1])
    x_max = int(box[2])
    y_max = int(box[3])

    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)

    x_min = max(0, x_min-int(0.2*bbox_height))
    y_min = max(0, y_min-int(0.2*bbox_width))
    x_max += int(0.2*bbox_height)
    y_max += int(0.2*bbox_width)

    hp = face['p_pred_deg']
    hy = face['y_pred_deg']
    hr = face['r_pred_deg']

    image = face['image']
    image = cv2.resize(image, (210, 70),
                        interpolation=cv2.INTER_CUBIC)
    image = transforms(image).to(device)
    # Check the devices of the inputs before the forward pass
    head_pos = torch.unsqueeze(torch.tensor(
        [float(hp), float(hr), float(hy)], dtype=torch.float32), dim=0).to(device)
    image = torch.unsqueeze(image, dim=0).to(device)
    res = net((image, head_pos))
    res = res.tolist()[0]
    pitch = res[0]
    yaw = -res[1]
    
    #print(pitch, yaw)

    draw_axis(affinity_frame, yaw, pitch, hr,
                    x_min+int(.5*(x_max-x_min)), y_min+int(.5*(y_max-y_min)), size=130)
    return

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    #x1 = size * (cos(yaw) * cos(roll)) + tdx
    #y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    #x2 = size * (-cos(yaw) * sin(roll)) + tdx
    #y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    #cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    #cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(250,150,0),8)

    return img
