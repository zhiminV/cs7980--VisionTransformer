# Bronte Sihan Li, 2024

import numpy as np
import torch
import tensorflow as tf
import logging
from typing import Literal
from preprocess_data import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    _get_features_dict,
    _clip_and_normalize,
    calculate_fire_change,
    random_crop,
)


class NextDayFireDataset(torch.utils.data.Dataset):
    """Next Day Fire dataset."""

    def __init__(
        self,
        tf_dataset: tf.data.Dataset,
        clip_normalize: bool = True,
        limit_features_list: list = None,
        use_change_mask: bool = False,
        sampling_method: Literal[
            'random_crop', 'center_crop', 'downsample', 'original'
        ] = 'random_crop',
        mode: Literal['train', 'val', 'test'] = 'train',
    ):
        self.tf_dataset = tf_dataset
        self.feature_description = _get_features_dict(
            64, INPUT_FEATURES + OUTPUT_FEATURES
        )
        self.mask_values = [0, 1]  # only 0 and 1 are valid mask values
        self.clip_normalize = clip_normalize
        self.use_change_mask = use_change_mask
        self.sampling_method = sampling_method
        self.mode = mode
        self.limit_features_list = limit_features_list
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return self.tf_dataset.reduce(0, lambda x, _: x + 1).numpy()

    def __getitem__(self, idx):
        item = next(iter(self.tf_dataset.skip(idx).take(1)))
        item = tf.io.parse_single_example(item, self.feature_description)
        target = item.pop('FireMask')
        # Get the change mask
        if self.use_change_mask:
            target = calculate_fire_change(item.get('PrevFireMask'), target)
        if self.limit_features_list:
            item = {key: item[key] for key in self.limit_features_list}
            # Clip and normalize features
            if self.clip_normalize:
                item = [
                    _clip_and_normalize(item.get(key), key)
                    for key in self.limit_features_list
                ]
            else:
                item = [item.get(key) for key in self.limit_features_list]

        if not self.use_change_mask:
            target = tf.cast(target, tf.float16).numpy()
            if self.mode == 'train':
                # convert to binary mask
                target = np.where(target > 0, 1, 0)
        target = np.expand_dims(target, axis=0)
        features = [tf.cast(x, tf.float16).numpy() for x in list(item)]
        item = np.stack(features, axis=0)

        if self.sampling_method == 'random_crop':
            item, target = random_crop(
                item,
                target,
            )
        elif self.sampling_method == 'center_crop':
            item = item[:, 16:-16, 16:-16]
            target = target[:, 16:-16, 16:-16]
        elif self.sampling_method == 'downsample':
            item = item[:, ::2, ::2]
            target = target[:, ::2, ::2]
        elif self.sampling_method == 'original':
            pass
        else:
            raise NotImplementedError

        # assert not np.any(np.isnan(item))
        # assert not np.any(np.isnan(target))

        return item, target


