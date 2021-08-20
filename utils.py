import os
import multiprocessing
from typing import Mapping

import cv2
from glob import glob


def video2png(video_path):
    dataset_name, person_id, clip_id, _ = video_path.split("data/")[-1].split("/")
    target_dir = f"data/vox/{person_id}/{clip_id}"

    if os.path.exists(target_dir):
        frame = len(os.listdir(target_dir))
    else:
        os.makedirs(target_dir)
        frame = 0

    assert len(os.listdir(target_dir)) == frame

    vidcap = cv2.VideoCapture(video_path)
    vid_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert vid_width == 224 and vid_height == 224, print(f"{target_dir} is not 224 * 224")

    vid_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(0, vid_length, 125):
        vidcap.set(cv2.CAP_PROP_FRAME_COUNT, frame_idx)
        ret, image = vidcap.read()

        cv2.imwrite(f"{target_dir}/{str(frame).zfill(4)}.jpg", image)
        if ret is False:
            break
        frame += 1

    print(f"{target_dir} Done")


def mp_video2png(n_processes):
    pool = multiprocessing.Pool(processes=n_processes)
    inputs = glob("data/vox_mp4/**/**/**.mp4")

    pool.map(video2png, inputs)
    pool.close()
    pool.join()


def resize(image):
    os.makedirs(image.replace("vox_128", "vox_256").replace(os.path.basename(image), ""), exist_ok=True)
    cv2.imwrite(image.replace("vox_128", "vox_256"), cv2.resize(cv2.imread(image), (256, 256), interpolation=cv2.INTER_CUBIC))
    print(image.replace("vox_128", "vox_256"))


def mp_resize(n_processes):
    pool = multiprocessing.Pool(processes=n_processes)
    inputs = glob("data/vox_128/**/**/**/**.jpg")

    pool.map(resize, inputs)
    pool.close()
    pool.join()



if __name__ == '__main__':
    mp_resize(12)