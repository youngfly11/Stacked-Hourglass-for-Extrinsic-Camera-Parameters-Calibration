#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019-04-28 11:46
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn

import os
import time
import torch
import pprint
import numpy as np
from tqdm import tqdm
from torch import nn
from lib.model import densenet, resnet
from lib.dataset.dataset_hg import get_image_dataloader
from lib.metric.losses import MeanSquareLoss
from lib.model.hourglass import hg
from lib.model.hourglass_gn import hg_gn
from lib.metric.jointMseLoss import JointsMSELoss
from lib.metric.pose_evaluation import final_preds
from lib.utils import get_inverse_images
from plus.meter import AverageValueMeter
from tensorboardX import SummaryWriter
import cv2
import os.path as osp
import os
import imageio

class Regressor(object):

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        logger.info(pprint.pformat(cfg, indent=4))

        ## using tensorboard to record the running loss and acc
        self.tbwriter = SummaryWriter(log_dir=self.cfg.rundir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_dataloader()
        self.build_network()
        self.build_meters()
        self.best_val_epoch = -1
        self.best_val_loss = 100
        self.val_loss_prev = 100
        self.best_train_epoch = -1
        self.best_train_loss = 100

    def build_dataloader(self):

        self.train_loader, self.val_loader, self.test_loader = get_image_dataloader(
            batch_size_train=self.cfg.train.batch_size,
            batch_size_val=self.cfg.val.batch_size,
            batch_size_test=self.cfg.test.batch_size
        )

    def build_network(self):

        if self.cfg.network.name == 'Hourglass':
            self.model = hg({'num_stacks':2, 'num_blocks':1, 'num_classes':1})
            # self.model = resnet.ResNetBackbone(pretrain=True, layers=self.cfg.network.layers)
        elif self.cfg.network.name == 'ResNet34':
            self.model = resnet.ResNetBackbone(pretrain=True, layers=self.cfg.network.layers)
        elif self.cfg.network.name == 'ResNet50':
            self.model = resnet.ResNetBackbone(pretrain=True, layers=self.cfg.network.layers)
        else:
            raise NotImplementedError
        self.build_optimizer()
        self.model = nn.DataParallel(self.model).cuda()

    def build_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(),
        lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        self.criterion = JointsMSELoss(use_target_weight=False).to(self.device)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.train.lr_decay_epochs, gamma=self.cfg.train.lr_decay_gamma)

    def build_meters(self):
        self.training_time_meter = AverageValueMeter()
        self.loss_meter_train = AverageValueMeter()
        self.loss_meter_val = AverageValueMeter()
        self.running_loss_meter_train = AverageValueMeter()
        self.running_loss_meter_val = AverageValueMeter()

    def reset_meters(self, type='train'):

        ## to recode the loss and acc within epoch
        if type == 'train':
            self.loss_meter_train.reset()
        elif type == 'val':
            self.loss_meter_val.reset()
        else:
            raise NotImplementedError

    def forward(self, dataloader, is_training=True, epoch=None):

        loss_mean = 0

        for step, data_sample in enumerate(dataloader):
            image, label = data_sample['image'], data_sample['label']
            line_det, edge_det = data_sample['line_det'], data_sample['edge_det']
            image = torch.cat((image, line_det, edge_det), 1)
            image, label = image.to(self.device), label.to(self.device).float()
            pred = self.model(image)

            loss = 0
            loss_ratio = [0.5, 1]
            for idx, stack_pred in enumerate(pred):
                loss += loss_ratio[idx]*self.criterion(output=stack_pred, target=label)

            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.loss_meter_train.add(loss.data.item())
                self.running_loss_meter_train.add(loss.data.item())
                loss_mean = self.loss_meter_train.value()[0]
                print(
                    'Train Epoch:{}, step:{}/{}, CurLoss:{:.8f}, RunningLoss:{:.8f}'.format(epoch, step, len(dataloader),
                                                                                      loss.data.item(),
                                                                                      self.running_loss_meter_train.value()[
                                                                                          0]))
                self.tbwriter.add_scalar('running_loss/training', self.running_loss_meter_train.value()[0],
                                         self.running_loss_meter_train.n)
            else:

                self.running_loss_meter_val.add(loss.data.item())
                self.tbwriter.add_scalar('running_loss/val', self.running_loss_meter_val.value()[0],
                                         self.running_loss_meter_val.n)
                self.loss_meter_val.add(loss.data.item())
                print(
                    'Val Epoch:{}, step:{}/{}, CurLoss:{:.8f}, RunningLoss:{:.8f}'.format(epoch, step, len(dataloader),
                                                                                      loss.data.item(),
                                                                                      self.running_loss_meter_val.value()[
                                                                                          0]))
                loss_mean = self.loss_meter_val.value()[0]

        print('Loss:{:.8f}'.format(loss_mean))

        return loss_mean

    def train(self):

        for epoch in range(self.cfg.train.start_epoch,
                           self.cfg.train.start_epoch + self.cfg.train.epochs):
            self.model.train()
            self.reset_meters(type='train')
            epoch_start_time = time.time()
            self.logger.info('Training Mode')

            loss_epoch_train = self.forward(self.train_loader, is_training=True, epoch=epoch)
            loss_epoch_val = self.val(epoch)

            self.tbwriter.add_scalar('loss_epoch_mean/loss_train_mean', loss_epoch_train, epoch)
            self.tbwriter.add_scalar('running_loss/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            print('Epoch: {}/{}, train_loss:{:.8f}, val_loss:{:.8f}, best_val_loss:{:.8f}, best_val_epoch:{:.8f}'.format(
                    epoch, self.cfg.train.epochs, loss_epoch_train, loss_epoch_val, self.best_val_loss, self.best_val_epoch))

            self.lr_scheduler.step()
            # if epoch == 10:
            #     for idx, param_group in enumerate(self.optimizer.param_groups):
            #         param_group['lr'] /= 10.0
            #
            # elif epoch > 10:
            #     if loss_epoch_val > self.val_loss_prev:
            #         for idx, param_group in enumerate(self.optimizer.param_groups):
            #             param_group['lr'] /= 10.0
            #         self.val_loss_prev = loss_epoch_val

            self._save_model(epoch, self.loss_meter_train.value()[0],
                             self.loss_meter_val.value()[0])

            # ETA
            epoch_end_time = time.time()
            self.training_time_meter.add(epoch_end_time - epoch_start_time)
            eta_time = (self.cfg.train.epochs - epoch) * self.training_time_meter.value()[0]
            print('ETA: {:.3f}mins'.format(eta_time / 60))

    def val(self, epoch):

        self.model.eval()
        self.reset_meters(type='val')
        self.logger.info('Validation Mode')
        with torch.no_grad():
            loss_epoch_val = self.forward(self.val_loader, is_training=False, epoch=epoch)
            self.tbwriter.add_scalar('loss_epoch_mean/loss_val_mean', loss_epoch_val, epoch)
        return loss_epoch_val

    def test(self):
        ckpt_name = 'checkpoint_best_val_loss'
        ckpt = torch.load(os.path.join(self.cfg.checkpointdir, ckpt_name + '.pth'))
        loss_train, loss_val, epoch = ckpt['train_loss'], ckpt['val_loss'], ckpt['epoch']
        log_str_1 = '\tTest: Name={}, train_loss={:.8f}, val_loss={:.8f}, Epoch={}\n'.format(ckpt_name, loss_train, loss_val, epoch)
        print(log_str_1)

        if not osp.exists(self.cfg.savevisdir):
            os.makedirs(self.cfg.savevisdir)
        self.model.load_state_dict(ckpt['state_dict'])
        self.test_loss_meter = AverageValueMeter()
        with torch.no_grad():
            self.model.eval()
            for step, data in enumerate(tqdm(self.test_loader)):
                image, label, label_coord = data['image'].to(self.device), data['label'].to(self.device).float(), data['label_coord']  # .cuda().float()
                pred = self.model(image)[-1]
                loss = self.criterion(pred, label)
                pred_coord = final_preds(output=pred.detach().cpu(), scale=4, res=[160, 120])
                image = get_inverse_images(image=image)
                label_coord = label_coord.numpy().astype(np.int32)
                pred_coord = pred_coord.astype(np.int32)
                for step_id in range(image.shape[0]):
                    image_per = image[step_id].copy()
                    label_step1 = label_coord[step_id]
                    pred_step1 = pred_coord[step_id]
                    image_per = cv2.circle(img=image_per, center=(label_step1[0], label_step1[1]), radius=5,color=(255,0,0), thickness=2)
                    image_per = cv2.circle(img=image_per, center=(pred_step1[0], pred_step1[1]), radius=5,color=(255,255,255), thickness=2)
                    imageio.imsave(osp.join(self.cfg.savevisdir, 'img_{}_{}.png'.format(step, step_id)), image_per)

                self.test_loss_meter.add(loss.data.item())
                print('Step:{}/{}, cur_loss:{:.8f}, running_loss:{:.8f}'.format(step, len(self.test_loader), loss.data.item(), self.test_loss_meter.value()[0]))
            print('Test_loss:{:.8f}'.format(self.test_loss_meter.value()[0]))

    def test_only(self):
        self.test()

    def _save_model(self, epoch, loss_train, loss_val):

        postfix='val_loss'

        if loss_val < self.best_val_loss:
            self.best_val_epoch = epoch
            self.best_val_loss = loss_val
            model_save_path = os.path.join(self.cfg.checkpointdir, 'checkpoint_best_{}.pth'.format(postfix))
            data_dict = {
                'state_dict': self.model.state_dict(),
                'train_loss': loss_train,
                'val_loss': loss_val,
                'epoch': epoch
            }
            torch.save(data_dict, model_save_path)
