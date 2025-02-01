from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
import gc
import time
import timm

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  False
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        #self.conv_stem = model.conv_stem
        #self.bn1 = model.bn1
        #self.act1 = model.act1
        self.conv_stem_1 = BasicConv(9, 32, kernel_size=3, stride=2, padding=1)

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])
    def forward(self, x):
        x1 = self.conv_stem_1(x)
        x2 = self.block0(x1)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]

class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att)*cv
        return cv

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv_guide1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=1,
                                             padding=0, stride=1, dilation=1))

        self.conv_guide2 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=1,
                                             padding=0, stride=1, dilation=1))

        self.agg = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.feature_att_8 = channelAtt(in_channels*2, 64)
        self.feature_att_16 = channelAtt(in_channels*4, 192)
        self.feature_att_up_8 = channelAtt(in_channels*2, 64)

    def forward(self, x, imgs, guide_volume01, guide_volume02):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2) + self.conv_guide2(guide_volume02)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)  +  self.conv_guide1(guide_volume01)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)

        return conv
        
class hourglass_light(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_light, self).__init__()

        self.conv0 = nn.Sequential(BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv1 = nn.Sequential(BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels*2, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),)


    def forward(self, x, volume_guided):
        conv0 = self.conv0(x) * volume_guided
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)
        conv = self.conv1_up(conv1)

        return [conv, conv1, conv2_up]     
        

class hourglass_att(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_att, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 

        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))


        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.feature_att_8 = channelAtt(in_channels*2, 64)
        self.feature_att_16 = channelAtt(in_channels*4, 192)
        self.feature_att_32 = channelAtt(in_channels*6, 160)
        self.feature_att_up_16 = channelAtt(in_channels*4, 192)
        self.feature_att_up_8 = channelAtt(in_channels*2, 64)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, imgs[3])

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)

        return conv


def pre_feature_pross(refimg_fea):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 3 * C, H, W])
    volume[:, 0:C, :, :]  = refimg_fea
    volume[:, C:2*C, 1:H, :]  = refimg_fea[:, :, 1:H, :] -  refimg_fea[:, :, :H-1, :]
    volume[:, 2*C:3*C, :, 1:W]  = refimg_fea[:, :, :, 1:W] -  refimg_fea[:, :, :, :W-1]

    volume = volume.contiguous()
    return volume

class SSANet_dist(nn.Module):
    def __init__(self, maxdisp, att_weights_only):
        super(SSANet_dist, self).__init__()
        self.maxdisp = maxdisp
        self.att_weights_only = att_weights_only
        self.feature = Feature()
        self.feature_up = FeatUp()
        chans = [16, 24, 32, 96, 160]

        self.stem_2_9 = nn.Sequential(
                      BasicConv(9, 32, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(32), nn.ReLU())
        self.stem_4 = nn.Sequential(
                      BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(48), nn.ReLU())
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(24, 32, True)
        self.spx_4 = nn.Sequential(
                     BasicConv(96, 24, kernel_size=3, stride=1, padding=1),
                     nn.Conv2d(24, 24, 3, 1, 1, bias=False),
                     nn.BatchNorm2d(24), nn.ReLU())
        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)

        self.desc_corr = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.desc_diff = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.conv_corr = nn.Sequential(
                              BasicConv(48, 96, kernel_size=3, stride=1, padding=1),
                              nn.Conv2d(96, 48, 3, 1, 1, bias=False))
        self.conv_diff = nn.Sequential(
                              BasicConv(48, 96, kernel_size=3, stride=1, padding=1),
                              nn.Conv2d(96, 48, 3, 1, 1, bias=False))        
        self.corr_stem_init = BasicConv(1, 1, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att_4 = channelAtt(8, 96)
        self.hourglass_att = hourglass_att(8)
    
 

    def forward(self, left, right):

        left_arg = pre_feature_pross(left)
        right_arg = pre_feature_pross(right)


        features_left = self.feature(left_arg)
        features_right = self.feature(right_arg)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2_9(left_arg)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2_9(right_arg)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)


        match_shift_left = self.conv(features_left[0])
        match_shift_right = self.conv(features_right[0])
        match_left = self.desc_corr(self.conv_corr(match_shift_left))
        match_right = self.desc_corr(self.conv_corr(match_shift_right))
        match_shift_left = self.desc_diff(self.conv_diff(match_shift_left))
        match_shift_right = self.desc_diff(self.conv_diff(match_shift_right))

        corr_volume = build_norm_correlation_volume(match_left, match_right, match_shift_left, match_shift_right, self.maxdisp//4)
        corr_volume_init_pred = self.corr_stem_init(corr_volume)
        corr_volume = self.corr_stem(corr_volume_init_pred)
        cost_att = self.corr_feature_att_4(corr_volume, features_left[0])
        
        att_weights = self.hourglass_att(cost_att, features_left)
        
        att_weights_prob = F.softmax(att_weights, dim=2)
        _, ind = att_weights_prob.sort(2, True)
        k = 24
        ind_k = ind[:, :, :k]
        ind_k = ind_k.sort(2, False)[0]
        
        att_topk = torch.gather(att_weights_prob, 2, ind_k)
        disparity_sample_topk = ind_k.squeeze(1).float()
               

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        if self.training:

            prob = att_weights_prob.squeeze(1)
            att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
            att_prob = F.softmax(att_prob, dim=1)
            pred_att = torch.sum(att_prob*disparity_sample_topk, dim=1)
            pred_att_up = context_upsample(pred_att.unsqueeze(1), spx_pred)
            
            att_prob_init = torch.gather(corr_volume_init_pred, 2, ind_k).squeeze(1)
            att_prob_init = F.softmax(att_prob_init, dim=1)
            pred_att_init = torch.sum(att_prob_init*disparity_sample_topk, dim=1)
            pred_att_up_init = context_upsample(pred_att_init.unsqueeze(1), spx_pred)
            
            mask = (torch.abs(pred_att_up - pred_att_up_init) > 2)
            

            return [pred_att_up*4, pred_att*4], mask



        else:

            att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
            att_prob = F.softmax(att_prob, dim=1)
            pred_att = torch.sum(att_prob*disparity_sample_topk, dim=1)
            pred_att_up = context_upsample(pred_att.unsqueeze(1), spx_pred)
            return [pred_att_up*4]

