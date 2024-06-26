# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS, build_dataset


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be ``times`` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (dict): The config of the dataset to be repeated.
        times (int): Repeat times.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, dataset, times, test_mode=False):
        dataset['test_mode'] = test_mode
        self.dataset = build_dataset(dataset)
        self.times = times
        if hasattr(dataset, 'class_prob'):
            self.class_prob = dataset.class_prob

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get data."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len