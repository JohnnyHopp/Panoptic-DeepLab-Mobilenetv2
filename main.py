import numpy as np
import torch
import random
import argparse
import os
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm
from fvcore.common.file_io import PathManager


from data.dataset import Dataset
from model.model_provider import create_model, create_optimizer
from evaluation.panoptic import CityscapesPanopticEvaluator
from evaluation.semantic_post_processing import get_semantic_segmentation
from evaluation.instance_post_processing import get_panoptic_segmentation
from evaluation.save_annotation import save_annotation, save_instance_annotation, save_panoptic_annotation
from utils.utils import AverageMeter,to_cuda

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()        
        self.parser.add_argument('-data', default='D:/programming/data/cityscapes', help='Input data folder')
        self.parser.add_argument('-saveDir', default='./save/result_RegularCE/epoch363/testdataset', help='Output data folder')
        self.parser.add_argument('-nThreads', default=0, type=int, help='Number of threads')
        self.parser.add_argument('-doAugmentaion', default=True, type=bool, help='To do augmentaion or not')
        self.parser.add_argument('-batchSize', default=1, type=int, help='Batch Size')
        self.parser.add_argument('-LR', default=1e-3, type=float, help='Learn Rate')
        self.parser.add_argument('-nEpoch', default=1000, type=int, help='Number of Epochs')
        self.parser.add_argument('-dropLR', default=10, type=float, help='Drop LR')
        self.parser.add_argument('-valInterval', default=1, type=int, help='Val Interval')
        self.parser.add_argument('-loadModel', default='./save/result_RegularCE/epoch363loss0.38450843513011934.pth', help='if not none, Load pre-trained model')
        self.parser.add_argument('-toTrain',  default=0, type=int, help='To train:1 or not:0')
        self.parser.add_argument('-toCuda',  default=1, type=int, help='To cuda:1 or not:0')
        self.parser.add_argument('-backBone', default='mobilenetV2', help='backbone network for encoder, mobilenetV2 or xception')
        self.parser.add_argument('-criterionSeg', default='RegularCE', help='criterion for semantic segmantation, RegularCE or DeepLabCE')
        self.opt = self.parser.parse_args()   

        #Create the saving directory and logging file for the configuration
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.saveDir):
            os.makedirs(self.opt.saveDir)
        file_name = os.path.join(self.opt.saveDir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        
def create_data_loaders(opt, split='train'):
    """
    Create the training data loader and test data loader
    """
    if split == 'train':
        tr_dataset  = Dataset(opt.data, opt, 'train')          
        train_loader = torch.utils.data.DataLoader(
            tr_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            drop_last=True,
            num_workers=opt.nThreads,
            pin_memory=True
        )
        return train_loader
    
    elif split == 'val':
        val_dataset = Dataset(opt.data, opt, 'val')       
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=opt.nThreads,
            pin_memory=True
        )
        return val_loader
    
    elif split == 'test':
        te_dataset = Dataset(opt.data, opt, 'test')
        test_loader = torch.utils.data.DataLoader(
            te_dataset,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=opt.nThreads,
            pin_memory=True
        )
        return test_loader        
    

def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    """
    To adjust the learning rate in training
    """
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def step(opt, data_loader, model, to_train=True, optimizer=None):
    """
    Used as a trining step or validation step
    """
    nIters = len(data_loader)
    loss_meter = AverageMeter()
    with tqdm(total=nIters) as t:
        for i, data in enumerate(data_loader):
            # ===================forward=====================
            if opt.toCuda:
                data = to_cuda(data, device())          
            image = data.pop('image')
            out_dict = model(image, data)
            loss = out_dict['loss']
            # ===================backward====================
            if to_train:                                             
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_meter.update(loss.detach().cpu().item(), image.size(0))
            t.set_postfix(loss='{:10.8f}'.format(loss_meter.avg))
            t.update()
        
    return loss_meter.avg


def train_net(opt, train_loader, test_loader, model, optimizer, n_epochs, val_interval, learn_rate, drop_lr):
    """
    To train the model with the input arguments, and save the plot of training and validation loss and accuracy, model parameters as well
    """
    loss_tr_list, loss_val_list = [], []
    for epoch in range(1,n_epochs+1):
        print('epoch',epoch)

        # ===================training====================
        model.train()
        loss_tr_avg = step(opt, train_loader, model, True, optimizer)
        # ===================validation====================
        model.eval()
        with torch.no_grad():
            loss_val_avg = step(opt, test_loader, model, False, optimizer)

        # ===================paramsSaving====================
        loss_tr_list.append(loss_tr_avg)
        loss_val_list.append(loss_val_avg)
        
        if loss_val_avg <= min(loss_val_list):
            torch.save(model.state_dict(), os.path.join(opt.saveDir,'epoch{}loss{}.pth'.format(epoch,loss_val_avg))) 

        # ===================lossPlotting====================
        plt.figure()
        plt.plot(loss_tr_list,label='trainLoss')
        plt.plot(loss_val_list,label='valLoss')            
        plt.legend(loc='upper left')

        plt.savefig(os.path.join(opt.saveDir, 'epoch{}.jpg'.format(epoch)))
        plt.close('all')
                              
#        adjust_learning_rate(optimizer, epoch, drop_lr, learn_rate)
        print('\n')
    
def device():
    """
    To put the Tensor or model in GPU if available
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    # Seed all sources of randomness to 0 for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0) if torch.cuda.is_available() else torch.manual_seed(0)
    random.seed(0)
    
    # Set cudnn.benchmark True to spped up training
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled =True
        torch.backends.cudnn.benchmark = True
    opt = Opts().opt

    # Create train and val data loaders
    train_loader, val_loader = create_data_loaders(opt, 'train'), create_data_loaders(opt, 'val')

    # Create nn
    model = create_model(opt)
    if opt.toCuda:
        model = model.to(device())
    # Choose to train or to test the model
    if opt.toTrain:
        # Create optimizer
        optimizer = create_optimizer(opt, model)
        train_net(opt, train_loader, val_loader, model, optimizer, opt.nEpoch, opt.valInterval, opt.LR, opt.dropLR)

    else:        
        # Change ASPP image pooling
        output_stride = 32
        train_crop_h, train_crop_w = (1025, 2049)
        scale = 1. / output_stride
        pool_h = int((float(train_crop_h) - 1.0) * scale + 1.0)
        pool_w = int((float(train_crop_w) - 1.0) * scale + 1.0)
    
        model.set_image_pooling((pool_h, pool_w))
    
        # Create test data loaders, change batch size to 1
        opt.batchSize = 1
        test_loader = create_data_loaders(opt, 'test')
        # test_loader = create_data_loaders(opt, 'test')

        panoptic_metric = CityscapesPanopticEvaluator(
            output_dir=os.path.join(opt.saveDir, 'panoptic'),
            train_id_to_eval_id=test_loader.dataset.train_id_to_eval_id(),
            label_divisor=test_loader.dataset.label_divisor,
            void_label=test_loader.dataset.label_divisor * test_loader.dataset.ignore_label,
            gt_dir=opt.data,
            split=test_loader.dataset.split,
            num_classes=test_loader.dataset.num_classes
        )

            
        image_filename_list = [
            os.path.splitext(os.path.basename(ann))[0] for ann in test_loader.dataset.img_list]

        debug_out_dir = os.path.join(opt.saveDir, 'debug_test')
        PathManager.mkdirs(debug_out_dir)
        
        model.eval()   
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if opt.toCuda:
                    data = to_cuda(data, device())
                image = data.pop('image')
                out_dict = model(image)
       
                # post-processing
                semantic_pred = get_semantic_segmentation(out_dict['semantic'])


                if 'foreground' in out_dict:
                    foreground_pred = get_semantic_segmentation(out_dict['foreground'])
                else:
                    foreground_pred = None
                                        
                panoptic_pred, center_pred = get_panoptic_segmentation(
                    semantic_pred,
                    out_dict['center'],
                    out_dict['offset'],
                    thing_list=test_loader.dataset.thing_list,
                    label_divisor=test_loader.dataset.label_divisor,
                    stuff_area=2048,
                    void_label=(
                            test_loader.dataset.label_divisor *
                            test_loader.dataset.ignore_label),
                    threshold=0.1,
                    nms_kernel=7,
                    top_k=200,
                    foreground_mask=foreground_pred)
                
                # save predictions
                semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()
    
                # Crop padded regions.
                image_size = data['size'].squeeze(0).cpu().numpy()
                panoptic_pred = panoptic_pred[:image_size[0], :image_size[1]]
    
                # Resize back to the raw image size.
                raw_image_size = data['raw_size'].squeeze(0).cpu().numpy()
                if raw_image_size[0] != image_size[0] or raw_image_size[1] != image_size[1]:
                    semantic_pred = cv2.resize(semantic_pred.astype(np.float), (raw_image_size[1], raw_image_size[0]),
                                               interpolation=cv2.INTER_NEAREST).astype(np.int32)
                    panoptic_pred = cv2.resize(panoptic_pred.astype(np.float),
                                               (raw_image_size[1], raw_image_size[0]),
                                               interpolation=cv2.INTER_NEAREST).astype(np.int32)
                
                # Optional: evaluates panoptic segmentation.
                image_id = '_'.join(image_filename_list[i].split('_')[:3])
                panoptic_metric.update(panoptic_pred,
                                       image_filename=image_filename_list[i],
                                       image_id=image_id)               


                # Processed outputs
#                save_annotation(semantic_pred, debug_out_dir, 'semantic_pred_%d' % i,
#                                add_colormap=True, colormap=test_loader.dataset.create_label_colormap())
#                pan_to_sem = panoptic_pred // test_loader.dataset.label_divisor
#                save_annotation(pan_to_sem, debug_out_dir, 'pan_to_sem_pred_%d' % i,
#                                add_colormap=True, colormap=test_loader.dataset.create_label_colormap())
#                ins_id = panoptic_pred % test_loader.dataset.label_divisor
#                pan_to_ins = panoptic_pred.copy()
#                pan_to_ins[ins_id == 0] = 0
#                save_instance_annotation(pan_to_ins, debug_out_dir, 'pan_to_ins_pred_%d' % i)

                save_panoptic_annotation(panoptic_pred, debug_out_dir, 'panoptic_pred_%d' % i,
                                         label_divisor=test_loader.dataset.label_divisor,
                                         colormap=test_loader.dataset.create_label_colormap())
            print('1111111111111111111111')
            results = panoptic_metric.evaluate()
            print(results)

      
       
if __name__ == '__main__':
    main()

