import torch
import face_alignment
from torchvision import transforms
from glob import glob
from modules.pretrained_models import Hopenet
import cv2
import numpy as np


if __name__ == '__main__':

    transform_hopenet = transforms.Compose([transforms.Resize(size=(224, 224)),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                      device='cpu')


    def get_head_pose(images):

        b, c, h, w = images.shape
        preds = fa.face_detector.detect_from_batch(images * 255)

        original_images = []
        for idx, pred in enumerate(preds):
            if len(pred) == 1:
                lt_x, lt_y, rb_x, rb_y, _ = pred[0]
                bw = rb_x - lt_x
                bh = rb_y - lt_y

                x_min = int(max(lt_x - 2 * bw / 4, 0))
                x_max = int(min(rb_x + 2 * bw / 4, w - 1))
                y_min = int(max(lt_y - 3 * bh / 4, 0))
                y_max = int(min(rb_y + bh / 4, h - 1))

                try:
                    img = images[idx, :, y_min:y_max, x_min:x_max]
                    original_images.append(img)
                except Exception as _:
                    img = images[idx]
                    original_images.append(img)
            else:
                img = images[idx]
                original_images.append(img)

        return original_images

    images = glob('**.jpg')
    input_images = []
    for idx, image in enumerate(images):
        img = cv2.resize(cv2.imread(image), (512, 512)).astype(np.float) / 255
        input_images.append(img)

    input_images = np.array(input_images)
    input_tensor = torch.from_numpy(input_images).permute(0, 3, 1, 2)

    aligned_images = get_head_pose(input_tensor)
    for idx, aimg in enumerate(aligned_images):
        cv2.imwrite(f'crop_{idx}.png', aimg.permute(1, 2, 0).numpy() * 255)