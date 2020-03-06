import os
import numpy as np
import torch
import torchvision.transforms as standard_transforms
import time
from PIL import Image
from torch.utils.data import DataLoader

from utils.tools import scale_process, evaluate, check_mkdir
import utils.transforms as extended_transforms
from dataset import Cityscapes as cityscapes
from models.AFENet import AFENet


# validate on val set
# single scale inference
# or multi-scale inference
# args should be changed with your data
args = {
    'snapshot': 'epoch_237_loss_0.11396_acc_0.96269_acc-cls_0.82723_mean-iu_0.74161_fwavacc_0.93174_lr_0.0011274619.pth',
    'model_save_path': './save_models/cityscapes',
    'dataset_path': '/xxxx/datasets/cityscapes',
    'scale': [0.75, 1.0, 1.25, 1.5, 1.75],  # or [1.0] with single scale inference
    'val_save_or_not': True,
    'val_exp_save_path': './val_exp_save_path',
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

    val_set = cityscapes.CityScapes(dataset_path, 'fine', 'val', transform=input_transform,
                                          target_transform=target_transform, val_scale=args['scale'])
    val_loader = DataLoader(val_set, batch_size=1, num_workers=1, shuffle=False)
    count = len(val_loader)
    validate(val_loader, net, input_transform, restore_transform, args['scale'], count)


def validate(t_val_loader, net, input_transform, restore, scales, count):
    net.eval()
    inputs_all, gts_all, predictions_all = [], [], []
    crop_size = 768
    for vi, (inputs, targets) in enumerate(t_val_loader):
        prediction = np.zeros((19, 1024, 2048), dtype=float)
        img = inputs.data[0]
        img = restore(img)

        inputs_all.append(img)

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

        prediction = prediction.reshape([1, 1024, 2048])

        gts_all.append(targets.data.cpu().numpy())
        predictions_all.append(prediction)
        if (vi + 1) % 20 == 0:
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('%s Completed %d / %d' % (localtime, vi+1, count))

    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(predictions_all)
    print(gts_all.shape)
    print(predictions_all.shape)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, cityscapes.num_classes)

    if args['val_save_or_not']:
        check_mkdir(args['val_exp_save_path'])
        for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
            if data[0] is None:
                continue
            input_pil = data[0]
            gt_pil = cityscapes.colorize_mask(data[1])
            predictions_pil = cityscapes.colorize_mask(data[2])
            input_pil.save(os.path.join(args['val_exp_save_path'], '%d_input.png' % idx))
            predictions_pil.save(os.path.join(args['val_exp_save_path'], '%d_prediction.png' % idx))
            gt_pil.save(os.path.join(args['val_exp_save_path'], '%d_gt.png' % idx))
    print('-----------------------------------------------------------------------------------------------------------')
    print('[acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (acc, acc_cls, mean_iu, fwavacc))

    print('-----------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
