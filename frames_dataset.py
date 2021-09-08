import os
from glob import glob
from skimage import io, img_as_float32

import dlib
import numpy as np
from torch.utils.data import Dataset


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - folder with frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True, random_seed=42, **kwargs):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')

        train_videos = []
        for identity in os.listdir(self.root_dir):
            train_videos += glob(f'{os.path.join(self.root_dir, identity)}/**')

        test_videos = []
        for identity in os.listdir(self.root_dir):
            test_videos += glob(f'{os.path.join(self.root_dir, identity)}/**')

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path = self.videos[idx]
        
        frames = os.listdir(path)
        num_frames = len(frames)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

        images = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        source = np.array(images[0], dtype='float32')
        driving = np.array(images[1], dtype='float32')

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
