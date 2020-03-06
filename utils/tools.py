import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image


def net_process(net, image_crop, crop_size, input_transform):
    image_crop = input_transform(image_crop).view(1, 3, crop_size, crop_size)  # 1 3 768 768   (1,3,1024,2048)

    image_crop = torch.cat([image_crop, image_crop.flip(3)], dim=0)
    image_crop = Variable(image_crop).cuda()
    output = net(image_crop)
    output = F.softmax(output, dim=1)
    output = (output[0] + output[1].flip(2)) / 2
    output = output.data.cpu().numpy()
    return output


def scale_process(net, scale_img, origin_w, origin_h, crop_size, img_w, img_h, input_transform):
    grid_w = int(np.ceil(img_w/float(crop_size)))
    grid_h = int(np.ceil(img_h/float(crop_size)))
    stride_w = int((img_w - crop_size) / (grid_w - 1))
    if grid_h == 1:
        stride_h = 0
    else:
        stride_h = int((img_h - crop_size) / (grid_h - 1))
    scale_prediction = np.zeros((19, img_h, img_w), dtype=float)
    count_crop = np.zeros((img_h, img_w), dtype=float)
    for index_h in range(0, grid_h):
        s_h = index_h * stride_h
        e_h = s_h + crop_size
        for index_w in range(0, grid_w):
            s_w = index_w * stride_w
            e_w = s_w + crop_size
            image_crop = scale_img.crop((s_w, s_h, e_w, e_h))
            count_crop[s_h:e_h, s_w:e_w] += 1
            scale_prediction[:, s_h:e_h, s_w:e_w] += net_process(net, image_crop, crop_size, input_transform)

    scale_prediction /= np.expand_dims(count_crop, axis=0)
    prediction = np.zeros((19, origin_h, origin_w), dtype=float)
    for i in range(19):
        prediction[i, :, :] = cv2.resize(scale_prediction[i, :, :], (origin_w, origin_h), interpolation=cv2.INTER_LINEAR)
    return prediction


def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_union(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
