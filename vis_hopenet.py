import os.path

import torch
import torchvision
from torchvision import transforms

import cv2
import dlib
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from math import cos, sin
from modules.hopenet import Hopenet


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


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
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


if __name__ == '__main__':

    # ResNet50 structure
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Load snapshot
    saved_state_dict = torch.load('pretrained/hopenet.pkl')
    model.load_state_dict(saved_state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    cnn_face_detector = dlib.cnn_face_detection_model_v1('pretrained/face_detector.dat')

    transformations = transforms.Compose([transforms.Scale(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    images = glob('data/khair/**.jpg')
    output_dir = 'output/khair/'
    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(images, total=len(images)):
        img = Image.open(img_path)
        cv2_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        dets = cnn_face_detector(cv2_img, 1)
        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(cv2_img.shape[1], x_max))
            y_max = int(min(cv2_img.shape[0], y_max))

            # Crop image
            img = cv2_img[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)

            img_tensor = transformations(img).unsqueeze(0).cuda()

            yaw, pitch, roll = model(img_tensor)

            # Continuous predictions
            yaw_predicted = softmax_temperature(yaw.data, 1)
            pitch_predicted = softmax_temperature(pitch.data, 1)
            roll_predicted = softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
            draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size=bbox_height/2)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), cv2_img)