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

def get_input_data(image, offset_coeff=1) -> dict:
    try:
        coeff = 1280 / image.shape[1]
        resized_image = cv2.resize(image, (1280, int(image.shape[0]*coeff)))
        with torch.no_grad():
            faces = detector(resized_image)
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

                c = cv2.waitKey(1)
                if c == 27:
                    break

                R_pred = model(img)

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi

                curr = {'p_pred_deg': euler[:, 0].cpu(),
                        'y_pred_deg': euler[:, 1].cpu(),
                        'r_pred_deg': euler[:, 2].cpu()
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
    except Exception as e:
        print(e.args)
        return None
    return result

def draw_eye_axis(img, yaw, pitch, roll, tdx, tdy, size=100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    x = size * (sin(yaw)) + tdx
    y = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x), int(y)), (255, 255, 0), 3)

    return img
