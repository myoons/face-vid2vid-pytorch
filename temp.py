import face_alignment
from glob import glob

import torch
from skimage import io
import cv2
import torchvision
import torch
import torch.nn.functional as F
from math import cos, sin
from PIL import Image
from modules.pretrained_models import Hopenet
import torchvision.transforms as transforms
import numpy as np


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
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

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


if __name__ == '__main__':

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    transform_hopenet = transforms.Compose([transforms.Resize(size=(224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])], )

    # ResNet50 structure
    model = Hopenet()
    model.eval()

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)

    for idx, p_img in enumerate(glob('data/vox-256/**/**/**/**.png')):
        print(p_img)
        image = io.imread(p_img)
        h, w, c = image.shape
        lt_x, lt_y, rb_x, rb_y, _ = fa.face_detector.detect_from_image(image)[0]

        bw = rb_x - lt_x
        bh = rb_y - lt_y

        x_min = int(max(lt_x - 2 * bw / 4, 0))
        x_max = int(min(rb_x + 2 * bw / 4, w))
        y_min = int(max(lt_y - 3 * bh / 4, 0))
        y_max = int(min(rb_y + bh / 4, h))

        cv2_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        print(img)
        input()
        img = Image.fromarray(img)

        print(img)
        input()
        img = transform_hopenet(img)
        print(img)
        input()
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

        yaw, pitch, roll = model(img)

        yaw_predicted = F.softmax(yaw, dim=1)
        pitch_predicted = F.softmax(pitch, dim=1)
        roll_predicted = F.softmax(roll, dim=1)

        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        canvas = cv2.imread(p_img)
        draw = draw_axis(canvas, yaw_predicted, pitch_predicted, roll_predicted,
                         tdx=(x_min + x_max) / 2,
                         tdy=(y_min + y_max) / 2,
                         size=bh / 2)

        print(yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item())
        cv2.imwrite(f'test.png', draw)
        input()
