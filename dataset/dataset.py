import os
from enum import Enum
from glob import glob
from typing import Tuple

import torch.utils.data
from torchvision.io import read_image


class DatasetType(Enum):
    TRAIN = 'train'
    EVAL_BALANCED = 'eval'
    EVAL_UNBALANCED = 'eval_unbalanced'


class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory: str, _type: DatasetType, transform=None, target_transform=None,
                 device: str = 'cpu'):
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = _type
        self.device = device

        match _type:
            case DatasetType.TRAIN:
                classes = []
                paths = sorted(glob(f'{os.path.join(data_directory, "train")}/*.jpg'),
                               key=lambda p: int(os.path.basename(p).split('.')[0]))
                labels = [0] * len(paths)

            case DatasetType.EVAL_BALANCED:
                classes, paths, labels = self.__get_labeled_data(data_directory, "test_balanced")

            case DatasetType.EVAL_UNBALANCED:
                classes, paths, labels = self.__get_labeled_data(data_directory, "test_unbalanced")

            case _:
                raise NotImplemented()

        self.data = paths
        self.labels = labels
        self.labels_legend = classes

    def __get_labeled_data(self, data_directory: str, sub_directory: str):
        classes = ['normal', 'anomaly']
        paths = []
        labels = []

        for c_ind, c_name in enumerate(classes):
            class_paths = sorted(glob(f'{os.path.join(data_directory, sub_directory, c_name)}/*.jpg'),
                                 key=lambda p: int(os.path.basename(p).split('.')[0]))
            paths += class_paths
            labels += [c_ind] * len(class_paths)

        return classes, paths, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, any]:
        img_path = self.data[idx]
        label = self.labels[idx]

        img = read_image(img_path).to(device=self.device)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label
