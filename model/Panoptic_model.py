import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

from collections import OrderedDict
from functools import partial


from .xception import xceptionNet, model_urls
from .mobilenet import MobileNetV2
from .aspp import ASPP
from .conv_module import stacked_conv

from .criterion import RegularCE, OhemCE, DeepLabCE
from utils.utils import AverageMeter
    
class Decoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, aspp_channels=None):
        super(Decoder, self).__init__()
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(in_channels, out_channels=aspp_channels, atrous_rates=atrous_rates)
        # self.aspp = ASPP(opt.aspp_in_channels, opt.aspp_out_channels, (6, 12, 18))
        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU()
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_conv(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        x = features[self.feature_key]
        x = self.aspp(x)

        # build decoder
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)

        return x


class Head(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(Head, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(
                    decoder_channels,
                    head_channels,
                ),
                nn.Conv2d(head_channels, num_classes[i], 1)
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred



class PanopticSegment(nn.Module):
    def __init__(self, opt):
        super(PanopticSegment, self).__init__()
        self.opt = opt
        if opt.criterionSeg == 'RegularCE':
            self.semantic_loss = RegularCE(ignore_label=255)
        elif opt.criterionSeg == 'DeepLabCE':
            self.semantic_loss = DeepLabCE(ignore_label=255, top_k_percent_pixels=0.15)
        self.semantic_loss_weight = 1.0
        self.loss_meter_dict = OrderedDict()
        self.loss_meter_dict['Loss'] = AverageMeter()
        self.loss_meter_dict['Semantic loss'] = AverageMeter()

        self.center_loss = nn.MSELoss(reduction='none')
        self.center_loss_weight = 200.0
        self.offset_loss = nn.L1Loss(reduction='none')
        self.offset_loss_weight = 0.01
        self.loss_meter_dict['Center loss'] = AverageMeter()
        self.loss_meter_dict['Offset loss'] = AverageMeter()
        
        if opt.backBone == 'xception':
            self.encoder = xceptionNet(self.opt)
            self.in_channels = 2048
            self.low_level_channels=(728, 256)
            self.sem_low_level_channels_project=(64, 32)
            self.ins_low_level_channels_project=(32, 16)
        elif opt.backBone == 'mobilenetV2':
            self.encoder = MobileNetV2()
            self.in_channels = 320
            self.low_level_channels=(32, 24)
            self.sem_low_level_channels_project=(64, 32)
            self.ins_low_level_channels_project=(32, 16)
        # Load the pretrained backbone params if there is no pretrained PanopticSegment params
        if opt.loadModel == 'none':
            if opt.backBone == 'xception':
#            state_dict = load_state_dict_from_url(model_urls['xception65'], progress=True)
                PATH = 'tf-xception65-270e81cf.pth'
            elif opt.backBone == 'mobilenetV2':
                PATH = 'mobilenet_v2-b0353104.pth'
            state_dict = torch.load(PATH, map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(state_dict, strict=False)  
        
        self.semantic_decoder = Decoder(in_channels=self.in_channels, 
                                        feature_key='res5', 
                                        low_level_channels=self.low_level_channels, 
                                        low_level_key=('res3', 'res2'), 
                                        low_level_channels_project=self.sem_low_level_channels_project,
                                        decoder_channels=256, 
                                        atrous_rates=(6, 12, 18), 
                                        aspp_channels=None)
        self.instance_decoder = Decoder(in_channels=self.in_channels, 
                                        feature_key='res5', 
                                        low_level_channels=self.low_level_channels, 
                                        low_level_key=('res3', 'res2'), 
                                        low_level_channels_project=self.ins_low_level_channels_project,
                                        decoder_channels=128, 
                                        atrous_rates=(6, 12, 18), 
                                        aspp_channels=256)
        
        self.semantic_head = Head(256, 256, [19], ['semantic'])
        self.instance_head = Head(128, 128, [1, 2], ['center', 'offset'])

        # Initialize decoder and head parameters.
        self._init_params(self.semantic_decoder)
        self._init_params(self.instance_decoder)
        self._init_params(self.semantic_head)
        self._init_params(self.instance_head)
        

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        self.instance_decoder.set_image_pooling(pool_size)

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.
            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result
                
    def forward(self, x, targets=None):
        pred = OrderedDict()
        
        input_shape = x.shape[-2:]
        features = self.encoder(x)
        
        # Semantic branch
        sem_pred = self.semantic_decoder(features)
        sem_pred = self.semantic_head(sem_pred)
        for key in sem_pred.keys():
            pred[key] = sem_pred[key]
            
            
        # Instance branch
        ins_pred = self.instance_decoder(features)
        ins_pred = self.instance_head(ins_pred)
        for key in ins_pred.keys():
            pred[key] = ins_pred[key]        

        results = self._upsample_predictions(pred, input_shape)
        
        if targets is None:
            return results
        else:
            return self.loss(results, targets)


    def loss(self, results, targets=None):
        batch_size = results['semantic'].size(0)
        loss = 0
        if targets is not None:
            if 'semantic_weights' in targets.keys():
                semantic_loss = self.semantic_loss(
                    results['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights']
                ) * self.semantic_loss_weight
            else:
                semantic_loss = self.semantic_loss(
                    results['semantic'], targets['semantic']) * self.semantic_loss_weight
            self.loss_meter_dict['Semantic loss'].update(semantic_loss.detach().cpu().item(), batch_size)
            loss += semantic_loss
            
            if self.center_loss is not None:
                # Pixel-wise loss weight
                center_loss_weights = targets['center_weights'][:, None, :, :].expand_as(results['center'])
                center_loss = self.center_loss(results['center'], targets['center']) * center_loss_weights
                # safe division
                if center_loss_weights.sum() > 0:
                    center_loss = center_loss.sum() / center_loss_weights.sum() * self.center_loss_weight
                else:
                    center_loss = center_loss.sum() * 0
                self.loss_meter_dict['Center loss'].update(center_loss.detach().cpu().item(), batch_size)
                loss += center_loss
                
            if self.offset_loss is not None:
                # Pixel-wise loss weight
                offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results['offset'])
                offset_loss = self.offset_loss(results['offset'], targets['offset']) * offset_loss_weights
                # safe division
                if offset_loss_weights.sum() > 0:
                    offset_loss = offset_loss.sum() / offset_loss_weights.sum() * self.offset_loss_weight
                else:
                    offset_loss = offset_loss.sum() * 0
                self.loss_meter_dict['Offset loss'].update(offset_loss.detach().cpu().item(), batch_size)
                loss += offset_loss
                
        # in train loop.
        results['loss'] = loss
        self.loss_meter_dict['Loss'].update(loss.detach().cpu().item(), batch_size)
        return results        
        
    def _init_params(self, model):
        # Backbone is already initialized (either from pre-trained checkpoint or random init).
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)