import os
from glob import glob
from skimage import io, img_as_float32
from skimage.transform import resize

import numpy as np
from torch.utils.data import Dataset


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - folder with frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True, random_seed=42, **kwargs):
        self.frame_shape = tuple(frame_shape)
        self.root_dir = root_dir  # /home/nas2_userF/dataset/Voxceleb2

        videos = []
        for folder in glob(f'{root_dir}/**'):
            videos += glob(f'{folder}/mp4_train/**')

        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        
        frames = glob(f'{video}/**.jpg')
        num_frames = len(frames)

        frame_idx = np.random.choice(num_frames, replace=False, size=2)

        images = [img_as_float32(resize(io.imread(frames[idx]), (self.frame_shape[0], self.frame_shape[1]))) for idx in frame_idx]
        source = np.array(images[0], dtype='float32')
        driving = np.array(images[1], dtype='float32')

        if source.shape != self.frame_shape:
            source = np.array(img_as_float32(resize(io.imread(frames[0]), (self.frame_shape[0], self.frame_shape[1]))), dtype='float32')
        if driving.shape != self.frame_shape:
            driving = np.array(img_as_float32(resize(io.imread(frames[-1]), (self.frame_shape[0], self.frame_shape[1]))), dtype='float32')
        
        return source.transpose((2, 0, 1)), driving.transpose((2, 0, 1))


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
