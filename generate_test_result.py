import os
import numpy as np
import time
from PIL import Image
import cv2
import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.tools import scale_process, check_makedirs
import utils.transforms as extended_transforms
from dataset import Cityscapes as cityscapes
from models.AFENet import AFENet

# generate test results on test set
# single scale inference
# or multi-scale inference
# args should be changed with your data
args = {
    'snapshot': 'epoch_237_loss_0.11396_acc_0.96269_acc-cls_0.82723_mean-iu_0.74161_fwavacc_0.93174_lr_0.0011274619.pth',
    'model_save_path': './save_models/cityscapes',
    'test_result_save_path': './test_result/cityscapes',
    'dataset_path': '/xxxx/datasets/cityscapes',
    'scale': [0.75, 1.0, 1.25, 1.5, 1.75]  # or [1.0] with single scale inference
}


def main():
    net = AFENet(classes=19, pretrained_model_path=None).cuda()
    net.load_state_dict(torch.load(os.path.join(args['model_save_path'], args['snapshot'])))

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()

    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
    dataset_path = args['dataset_path']

    test_set = cityscapes.CityScapes(dataset_path, 'fine', 'test', transform=input_transform,
                                          target_transform=target_transform, val_scale=args['scale'])
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)
    test(test_loader, net, input_transform, restore_transform, args['scale'])


def test(test_loader, net, input_transform, restore, scales):
    print('Start...')
    net.eval()
    trainid_to_id = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26,
                     14: 27, 15: 28, 16: 31, 17: 32, 18: 33}
    to_save_dir = args['test_result_save_path']
    check_makedirs(to_save_dir)

    crop_size = 768
    count = len(test_loader)
    for vi, (inputs, img_name) in enumerate(test_loader):
        prediction = np.zeros((19, 1024, 2048), dtype=float)
        img = inputs.data[0]
        img = restore(img)
        origin_w, origin_h = img.size
        for scale in scales:
            new_w = int(origin_w * scale)
            new_h = int(origin_h * scale)
            if scale == 1.0:
                scale_img = img
            else:
                scale_img = img.resize((new_w, new_h), Image.BILINEAR)
            prediction += scale_process(net, scale_img, origin_w, origin_h, crop_size, new_w, new_h, input_transform)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=0)
        prediction = prediction.reshape([1024, 2048])

        img_name = img_name[0]
        pred_copy = prediction.copy()
        for k, v in trainid_to_id.items():
            pred_copy[prediction == k] = v

        prediction = Image.fromarray(pred_copy.astype(np.uint8))
        prediction.save(os.path.join(to_save_dir, img_name))
        if (vi + 1) % 50 == 0:
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[%s] [iter %d / %d]' % (localtime, vi + 1, count))

    print('-----------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
