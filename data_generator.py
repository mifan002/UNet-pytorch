import imgaug.augmenters as iaa
import os.path
import numpy as np
import imageio as io
from skimage.transform import resize
import albumentations as A
import torch as t


class DataGenerator:
    """DataGenerator to generate 1 batch at a time.

    1.WHERE TO USE: should be called in train.py, for all epochs creating once should be enough.

    2.WHAT TO DO: load data and labels from a given directory. After encapsulated as an iterator(i.e. iter(DataGenerator)), can return 1 batch of data and labels at a time,
    functions like shuffling after each epoch, augmentation are included in the DataGenerator.

    Attributes:
        current_batch_idx: the batch-idx of the current call.
        one_epoch_end:

    """
    def __init__(self,image_ids, mask_ids, batch_size, num_class, height=224, width=224, augment=True, shuffle=True):
        """create a DataGenerator, with it batches of images and labels can be generated.

        :param image_ids: a list that contains paths to all images.
        :param mask_ids: a list that contains paths to masks of all images.
        :param batch_size: int, num of images/masks contained in 1 batch.
        :param num_class: int, num of classes in the mask, e.g. binary classification->2
        :param height: int, the height of to be generated images/masks.
        :param width: int, the width of to be generated images/masks.
        :param augment: bool, True if augmentation is needed before images/masks are generated.
        :param shuffle: bool, True if shuffling is needed after each epoch.
        """
        self.image_ids = image_ids
        self.mask_ids = mask_ids
        self.batch_size = batch_size
        self.num_class = num_class
        self.width = width
        self.height = height
        self.shuffle = shuffle
        self.augment = augment

        # used in __next__():
        # 1. represents the batch-idx of the current call
        # 2. counting how many batches are returned by __next__()
        self.current_batch_idx = 0

    def __len__(self):
        """
        :return: the num of batches contained within 1 epoch (rounded to floor)
        """
        return len(self.image_ids) // self.batch_size

    def on_epoch_end(self):
        """when 1 epoch done, do shuffling to the dataset, and reset current_batch_idx to 0 such that
        the DateGenerator can be reused for the next epoch without raising StopIteration error(when calling __next__()).

        :return: None
        """
        if self.shuffle:
            paired_ids = list(zip(self.image_ids, self.mask_ids))
            np.random.shuffle(paired_ids)
            self.image_ids, self.mask_ids = zip(*paired_ids)
            self.current_batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        one_batch_imgs = []
        one_batch_masks = []

        # picking out 1 batch of imgs_ids and masks_ids from the dataset
        idx_start = self.current_batch_idx * self.batch_size
        idx_end = (self.current_batch_idx + 1) * self.batch_size
        one_batch_imgs_ids = self.image_ids[idx_start:idx_end]
        one_batch_masks_ids = self.mask_ids[idx_start:idx_end]

        # load each img from its id(i.e. storing path) and do augmentation
        for one_img_id, one_mask_id in zip(one_batch_imgs_ids, one_batch_masks_ids):
            # Load image
            one_image = io.imread(one_img_id, as_gray=True)
            one_image = resize(one_image, (self.height, self.width))
            # normalize values from [0, 255] to [-1, 1] for faster convergence of the NN
            one_image = np.array(one_image) / 127.5 - 1
            # add the channel dimension, e.g. (224,224)->(224,224,1)
            one_image = one_image[None, ...]

            # Load mask
            one_mask = io.imread(one_mask_id, as_gray=True)
            one_mask = resize(one_mask, (self.height, self.width))
            one_mask = np.array(one_mask) / 255
            one_mask = one_mask[None, ...]

            # Data augmentation
            if self.augment:
                aug = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, border_mode=0, value=[0, 0, 0], p=0.5),
                    A.Blur(p=0.5)
                ])
                augmented = aug(image=one_image, mask=one_mask)
                one_image_aug = augmented['image']
                one_mask_aug = augmented['mask']

            one_batch_imgs.append(one_image_aug)
            one_batch_masks.append(one_mask_aug)

        one_batch_imgs = np.array(one_batch_imgs)
        one_batch_masks = np.round(np.array(one_batch_masks))


        if self.current_batch_idx >= self.__len__():
            raise StopIteration
        else:
            # change the counting
            self.current_batch_idx += 1

        return t.tensor(one_batch_imgs), t.tensor(one_batch_masks)

