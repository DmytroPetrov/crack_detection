import os
from enum import Enum
from glob import glob
from typing import Tuple

import pandas as pd
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


class SubSampCrackDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory: str, df: pd.DataFrame, _type: DatasetType, sub_transform=None, maj_transform=None,
                 device: str = 'cpu'):
        self.sub_transform = sub_transform
        self.maj_transform = maj_transform
        self.dataset_type = _type
        self.device = device

        match _type:
            case DatasetType.TRAIN:
                paths = sorted(glob(f'{os.path.join(data_directory, "train")}/*.jpg'),
                               key=lambda p: int(os.path.basename(p).split('.')[0]))
                p_df = pd.DataFrame(paths, columns=['path'])
                p_df['name'] = p_df['path'].apply(lambda p: int(os.path.basename(p).split('.')[0]))

                data_df = pd.merge(df, p_df, left_on='name', right_on='name')

                paths = data_df['path'].to_list()
                labels = data_df['anomaly'].to_list()
            case _:
                raise NotImplemented()

        self.df = data_df
        self.data = paths
        self.labels = labels
        self.labels_legend = ['normal', 'anomaly']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data[idx]
        label = self.labels[idx]

        img = read_image(img_path).to(device=self.device)

        if self.sub_transform and label == 1:
            img = self.sub_transform(img)
        if self.maj_transform and label == 0:
            img = self.maj_transform(img)

        return img, label
