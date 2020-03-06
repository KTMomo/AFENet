import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms

from utils.tools import scale_process
from dataset import Cityscapes as cityscapes
from models.AFENet import AFENet

# generate test results on test set
# single scale inference
# or multi-scale inference
# args should be changed with your data
args = {
    'snapshot': 'epoch_237_loss_0.11396_acc_0.96269_acc-cls_0.82723_mean-iu_0.74161_fwavacc_0.93174_lr_0.0011274619.pth',
    'model_save_path': './save_models/cityscapes',
    'test_result_save_path': './test_pred.png',
    'image_path': './test_img.png',
    'scale': [0.75, 1.0, 1.25, 1.5, 1.75]  # or [1.0] with single scale inference
}


def main():
    net = AFENet(classes=19, pretrained_model_path=None).cuda()
    net.load_state_dict(torch.load(os.path.join(args['model_save_path'], args['snapshot'])))
    net.eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    img_path = args['image_path']
    result_save_path = args['test_result_save_path']
    scales = args['scale']
    img = Image.open(img_path).convert('RGB')

    crop_size = 768
    prediction = np.zeros((19, 1024, 2048), dtype=float)
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

    prediction = cityscapes.colorize_mask(prediction)
    prediction.save(os.path.join(result_save_path))
    print("Completed.")


if __name__ == '__main__':
    main()
