import torch
import torch.nn as nn
from torchvision import models

from lib.rpn_util import *
from models.CA import *
from models import resnet

import numpy as np

class RPN(nn.Module):

    def __init__(self, phase, conf):
        super(RPN, self).__init__()
        train = phase.lower() == 'train'

        self.base = resnet.ResNetDilate(num_layer=50)
        self.depthnet = resnet.ResNetDilate(num_layer=50)

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]
        self.prev_num = 1 if conf.prev_num is None else conf.prev_num
        self.dropout_rate = 0.2 if conf.dropout_rate is None else conf.dropout_rate

        self.prop_feats = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.channel_attention_1 = ResidualGroup(
                RCAB,
                n_resblocks=12,
                n_feat=512,
                kernel_size=3,
                reduction=16, 
                act=nn.LeakyReLU(0.2, True), 
                norm=False)
        
        self.channel_attention_2 = ResidualGroup(
                RCAB,
                n_resblocks=12,
                n_feat=512,
                kernel_size=3,
                reduction=16, 
                act=nn.LeakyReLU(0.2, True), 
                norm=False)

        self.channel_attention_3 = ResidualGroup(
                RCAB,
                n_resblocks=12,
                n_feat=512,
                kernel_size=3,
                reduction=16, 
                act=nn.LeakyReLU(0.2, True), 
                norm=False)

        self.prop_feats_multi = nn.Sequential(
            nn.Conv2d(2048*3, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.dropout_channel = nn.Dropout2d(p=0.3)

        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1)

        # motion
        self.motion_x = nn.Conv2d(self.prop_feats_multi[0].out_channels, self.num_anchors, 1)
        self.motion_y = nn.Conv2d(self.prop_feats_multi[0].out_channels, self.num_anchors, 1)
        self.motion_z = nn.Conv2d(self.prop_feats_multi[0].out_channels, self.num_anchors, 1)

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors

        motion_scale = np.load('./lib/anchor_motion_scale_multi_objects_split1.npy')
        self.motion_scale = torch.from_numpy(motion_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        batch_size = x.size(0)
        curr_im = x[:, 0:3, :, :]
        prev_im = x[:, 3:6, :, :]
        depth = x[:, 6:7, :, :]
        depth = depth.repeat(1, 3, 1, 1)
        
        curr_x = self.base(curr_im)
        prev_x = self.base(prev_im)
        depth_x = self.depthnet(depth)

        x_single = curr_x
        x_multi = torch.cat([curr_x, prev_x, depth_x], dim=1) # [N, 1024*3, 32, 110]

        # multi-feature
        x_multi_in = self.prop_feats_multi(x_multi)

        x_multi = self.channel_attention_1(x_multi_in)
        x_multi = self.channel_attention_2(x_multi)
        prop_feats = self.channel_attention_3(x_multi)

        prop_feats = prop_feats + x_multi_in
        # dropout
        prop_feats = self.dropout(prop_feats)

        cls = self.cls(prop_feats)

        # motion
        self.motion_scale = self.motion_scale.type(torch.cuda.FloatTensor)
        motion_x = self.motion_x(prop_feats)
        motion_x = motion_x * self.motion_scale
        motion_y = self.motion_y(prop_feats)
        motion_z = self.motion_z(prop_feats)
        motion_z = motion_z * self.motion_scale

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)

        bbox_z3d = self.bbox_z3d(prop_feats)
        
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        motion_x = flatten_tensor(motion_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        motion_y = flatten_tensor(motion_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        motion_z = flatten_tensor(motion_z.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        motion = torch.cat([motion_x, motion_y, motion_z], dim=2)

        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        
        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2)

        # feat_size = [feat_h, feat_w]
        feat_size = torch.tensor([feat_h, feat_w])

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size, motion

        else:

            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
                self.rois = self.rois.type(torch.cuda.FloatTensor)

            return cls, prob, bbox_2d, bbox_3d, feat_size, motion, self.rois


def build(conf, phase='train'):

    train = phase.lower() == 'train'

    rpn_net = RPN(phase, conf)

    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net
