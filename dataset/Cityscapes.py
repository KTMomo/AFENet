import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

num_classes = 19
ignore_label = 255
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


class CityScapes(data.Dataset):
    def __init__(self, root, quality, mode, transform=None,
                 target_transform=None, crop_size=768, val_scale=None):
        if crop_size < 0 or crop_size > 1024:
            raise ValueError('The crop_size must be in (0,1024]')
        self.crop_size = crop_size
        self.imgs = self._make_dataset(root, quality, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.quality = quality
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.val_scale = val_scale
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        # origin id to train_id
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.mode == 'train':
            img, mask = self._train_transform(img, mask, self.crop_size)
        elif self.mode == 'val':
            if self.val_scale is None:
                img, mask = self._val_transform(img, mask, self.crop_size)  # To verify during testing
            else:
                img, mask = img, mask  # Original size: for single scale or multi-scale verification

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.mode == 'test':
            img_name = img_path.split('/')[-1]
            img_name = img_name.split('_leftImg8bit.png')[0]
            img_name = img_name + '*.png'
            return img, img_name

        return img, mask

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def _make_dataset(root, quality, mode):
        assert (quality == 'fine' and mode in ['train', 'val', 'test']) or \
               (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

        if quality == 'coarse':
            img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
            mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
            mask_postfix = '_gtCoarse_labelIds.png'
        else:
            mask_path = os.path.join(root, 'gtFine', mode)
            mask_postfix = '_gtFine_labelIds.png'

        img_path = os.path.join(root, 'leftImg8bit', mode)
        print(img_path)
        assert os.listdir(img_path) == os.listdir(mask_path)
        items = []
        categories = os.listdir(img_path)
        for c in categories:
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
            for it in c_items:
                item = (os.path.join(img_path, c, it + '_leftImg8bit.png'),
                        os.path.join(mask_path, c, it + mask_postfix))
                items.append(item)

        return items

    @staticmethod
    def _val_transform(img, mask, crop_size):
        w, h = img.size
        x1 = int((w - crop_size) / 2)
        y1 = int((h - crop_size) / 2)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        return img, mask

    @staticmethod
    def _train_transform(img, mask, crop_size):
        # random scale [0.75, 2]
        w, h = img.size  # (2048,1024)
        scale = 0.75 + 1.25 * random.random()
        oh = int(np.ceil(scale*h))
        ow = int(2 * oh)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # random crop crop_size
        x1 = random.randint(0, ow - crop_size)
        y1 = random.randint(0, oh - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # random flip left_right
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask
