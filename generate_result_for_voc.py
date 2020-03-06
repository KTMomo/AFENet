import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader

import utils.transforms as transform
from utils.tools import colorize, check_makedirs, AverageMeter, intersection_union
from dataset import VOC2012 as voc2012
from models.AFENet import AFENet

########################################################################
# args: dataset_root
#       ---The location where the VOC2012 dataset is decompressed.
#       ---dataset_root includes:
#                                JPEGImages
#                                SegmentationClassAug
#                                train.txt    test.txt  val.txt
#                                train_aug.txt  trainval.txt  trainval_aug.txt
#
########################################################################

args = {
    'snapshot': 'epoch_46_loss_0.23245_acc_0.93675_acc-cls_0.83493_mean-iu_0.75424_fwavacc_0.88422_lr_0.0010300419.pth',
    'model_save_path': './save_models/voc',
    'test_result_save_path': './test_result/voc_test/results/VOC2012/Segmentation/comp5_test_cls/',
    'scales': [0.75, 1.0, 1.25, 1.5, 1.75],  # or [1.0] with single scale inference

    'dataset_root': '/xxxx/datasets/VOC2012AUG',
    'test_list': '/xxxx/datasets/VOC2012AUG/test.txt',
    'colors_path': './dataset/voc2012_colors.txt'
}


def main():
    net = AFENet(classes=21, pretrained_model_path=None).cuda()
    net.load_state_dict(torch.load(os.path.join(args['model_save_path'], args['snapshot'])))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    test_transform = transform.Compose([transform.ToTensor()])

    test_data = voc2012.VOC2012(split='test', data_root=args['dataset_root'], data_list=args['test_list'],
                                 transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    gray_folder = os.path.join(args['test_result_save_path'])
    # gray_folder = os.path.join(args['test_result_save_path'], 'gray')
    color_folder = os.path.join(args['test_result_save_path'], 'color')

    colors = np.loadtxt(args['colors_path']).astype('uint8')

    test(test_loader, test_data.data_list, net, 21, mean, std, 512, 480, 480, args['scales'],
         gray_folder, color_folder, colors)

    # cal_acc(test_data.data_list, gray_folder, 21)


def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors):
    print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()
    count = len(test_loader)
    for i, (input, _) in enumerate(test_loader):
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)

        if ((i + 1) % 20 == 0) or (i + 1 == len(test_loader)):
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[%s] [iter %d / %d]' % (localtime, i + 1, count))

        check_makedirs(gray_folder)
        # check_makedirs(color_folder)

        gray = np.uint8(prediction)
        # color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        # color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        # color.save(color_path)
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        intersection, union, target = intersection_union(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        # print('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))


if __name__ == '__main__':
    main()
