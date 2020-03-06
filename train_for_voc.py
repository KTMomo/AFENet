import os
import random
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.tools import evaluate, check_makedirs
import utils.transforms as transform
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
    'train_batch_size': 8,
    'epoch_num': 50,
    'lr': 1e-2,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'val_batch_size': 8,
    'model_save_path': './save_models/voc',

    'dataset_root': '/xxxx/datasets/VOC2012AUG',
    'train_list': '/xxxx/datasets/VOC2012AUG/train_aug.txt',
    'val_list': '/xxxx/datasets/VOC2012AUG/val.txt',
    'pretrained_model_path': './models/initmodel/resnet101_v2.pth'
}


def main():
    net = AFENet(classes=21, pretrained_model_path=args['pretrained_model_path']).cuda()
    net_ori = [net.layer0, net.layer1, net.layer2, net.layer3, net.layer4]
    net_new = [net.ppm, net.cls, net.aux, net.ppm_reduce, net.aff1, net.aff2, net.aff3]

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([0.75, 2.0]),
        transform.RandomHorizontalFlip(),
        transform.Crop([480, 480], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    train_data = voc2012.VOC2012(split='train', data_root=args['dataset_root'], data_list=args['train_list'],
                                 transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=args['train_batch_size'], shuffle=True,
                                               num_workers=8, drop_last=True)

    val_transform = transform.Compose([
        transform.Crop([480, 480], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = voc2012.VOC2012(split='val', data_root=args['dataset_root'], data_list=args['val_list'],
                               transform=val_transform)

    val_loader = DataLoader(val_data, batch_size=args['val_batch_size'], shuffle=False, num_workers=8)

    if len(args['snapshot']) == 0:
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args['model_save_path'], args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                               'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                               'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}
    params_list = []
    for module in net_ori:
        params_list.append(dict(params=module.parameters(), lr=args['lr']))
    for module in net_new:
        params_list.append(dict(params=module.parameters(), lr=args['lr'] * 10))
    args['index_split'] = 5

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    optimizer = torch.optim.SGD(params_list,
                                lr=args['lr'],
                                momentum=args['momentum'],
                                weight_decay=args['weight_decay'])
    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(args['model_save_path'], 'opt_' + args['snapshot'])))

    check_makedirs(args['model_save_path'])

    all_iter = args['epoch_num']*len(train_loader)
    for epoch in range(curr_epoch, args['epoch_num'] + 1):
        train(train_loader, net, optimizer, epoch, all_iter)
        validate(val_loader, net, criterion, optimizer, epoch)


def train(train_loader, net, optimizer, epoch, all_iter, accumulation_steps=2):
    net.train()
    train_loss = 0.0
    curr_iter = (epoch - 1) * len(train_loader)

    for i, (inputs, targets) in enumerate(train_loader):
        assert inputs.size()[2:] == targets.size()[1:]
        inputs = Variable(inputs).cuda()
        targets = Variable(targets).cuda()

        _, main_loss, aux_loss = net(inputs, targets)
        loss = (main_loss + 0.4 * aux_loss) / accumulation_steps
        train_loss += loss.item()
        loss.backward()
        # real batch_size = arg['train_batch_size'] * 4
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            current_lr = args['lr'] * (1 - float(curr_iter) / all_iter)**0.9
            for index in range(0, args['index_split']):
                optimizer.param_groups[index]['lr'] = current_lr
            for index in range(args['index_split'], len(optimizer.param_groups)):
                optimizer.param_groups[index]['lr'] = current_lr * 10

        curr_iter += 1

        if (i + 1) % 100 == 0:
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('[%s], [epoch %d], [iter %d / %d], [train loss %.5f]' % (
                localtime, epoch, i + 1, len(train_loader), train_loss / (i + 1)
            ))


def validate(val_loader, net, criterion, optimizer, epoch):
    net.eval()
    val_loss = 0.0
    gts_all, predictions_all = [], []
    count = len(val_loader)
    with torch.no_grad():
        for vi, (inputs, targets) in enumerate(val_loader):
            inputs = Variable(inputs).cuda()
            targets = Variable(targets).cuda()
            outputs = net(inputs)

            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            val_loss += criterion(outputs, targets).item()

            gts_all.append(targets.data.cpu().numpy())
            predictions_all.append(predictions)

        gts_all = np.concatenate(gts_all)
        predictions_all = np.concatenate(predictions_all)

        acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, 21)

        if mean_iu > args['best_record']['mean_iu']:
            args['best_record']['val_loss'] = val_loss / count
            args['best_record']['epoch'] = epoch
            args['best_record']['acc'] = acc
            args['best_record']['acc_cls'] = acc_cls
            args['best_record']['mean_iu'] = mean_iu
            args['best_record']['fwavacc'] = fwavacc
            snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
                epoch, val_loss / count, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
            )

            torch.save(net.state_dict(), os.path.join(args['model_save_path'], snapshot_name + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args['model_save_path'], 'opt_' + snapshot_name + '.pth'))

        print('-----------------------------------------------------------------------------------------------------------')
        print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [lr %.10f]' % (
            epoch, val_loss / count, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']))

        print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
            args['best_record']['val_loss'], args['best_record']['acc'], args['best_record']['acc_cls'],
            args['best_record']['mean_iu'], args['best_record']['fwavacc'], args['best_record']['epoch']))

        print('-----------------------------------------------------------------------------------------------------------')

    return val_loss


if __name__ == '__main__':
    main()
