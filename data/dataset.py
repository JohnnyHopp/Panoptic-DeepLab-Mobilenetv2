import os
import cv2
import numpy as np
import glob
import json
import os
from PIL import Image, ImageOps

import torch
from torchvision import transforms
import torch.nn.functional as F

from . import transforms as T
from .target_transforms import PanopticTargetGenerator, SemanticTargetGenerator

# CITYSCAPES IMAGE CONSTANTS(RGB order)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

splits_to_sizes={'train': 2975,
                 'trainval': 3475,
                 'val': 500,
                 'test': 1525}
num_classes=19
ignore_label=255

# Add 1 void label.
_CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                            23, 24, 25, 26, 27, 28, 31, 32, 33, 0]

_CITYSCAPES_THING_LIST = [11, 12, 13, 14, 15, 16, 17, 18]

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}

class Dataset(torch.utils.data.Dataset):
    """
    Create the training dataset and test dataset, based on the data_path and the split of "train","val" or "test"
    """
    def __init__(self, root, opt, split='train'):
        self.root = root
        self.opt = opt
        self.split = split
        self.to_train = opt.toTrain
        self.is_train = split=='train'
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.label_divisor = 1000
        self.label_dtype = np.float32
        self.thing_list = _CITYSCAPES_THING_LIST


        
        # Get image and annotation list.
        if split == 'test':
            self.img_list = self._get_files('image', self.split)
            self.ann_list = None
            self.ins_list = None
        else:
            self.img_list = []
            self.ann_list = []
            self.ins_list = []
            #json file below is generated by cityscapesScripts, url='https://github.com/mcordts/cityscapesScripts'
            json_filename = os.path.join(self.root, 'gtFine', 'cityscapes_panoptic_{}_trainId.json'.format(self.split))
            dataset = json.load(open(json_filename))
            for img in dataset['images']:
                img_file_name = img['file_name']
                self.img_list.append(os.path.join(
                    self.root, 'leftImg8bit', self.split, img_file_name.split('_')[0],
                    img_file_name.replace('_gtFine', '')))
            for ann in dataset['annotations']:
                ann_file_name = ann['file_name']
                self.ann_list.append(os.path.join(
                    self.root, 'gtFine', 'cityscapes_panoptic_{}_trainId'.format(self.split), ann_file_name))
                self.ins_list.append(ann['segments_info'])

        assert len(self) == splits_to_sizes[self.split]

        self.pre_augmentation_transform = None
        self.target_transform = PanopticTargetGenerator(self.ignore_label, self.rgb2id, _CITYSCAPES_THING_LIST,
                                                        sigma=8, ignore_stuff_in_offset=True,
                                                        small_instance_area=0,
                                                        small_instance_weight=1)
        # Generates semantic label for evaluation.
        self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

                     
        self.do_augment = (self.is_train and opt.doAugmentaion)

        if self.do_augment:
            #Do normalization and augmentation on the data
            min_scale = 0.5
            max_scale = 2.0
            crop_h, crop_w = (513, 1025)
            scale_step_size = 0.1
            pad_value = tuple([int(v * 255) for v in MEAN])
            ignore = (0, 0, 0)
            flip_prob = 0.5
            mean = MEAN
            std = STD
        else:
            # no data augmentation
            min_scale = 1
            max_scale = 1
            crop_h, crop_w = (1025, 2049)
            scale_step_size = 0
            flip_prob = 0
            pad_value = tuple([int(v * 255) for v in MEAN])
            ignore = (0, 0, 0)
            mean = MEAN
            std = STD
        #Build transforms
        if split != 'test':
            self.transform = T.Compose(
                [
                    T.RandomScale(
                        min_scale,
                        max_scale,
                        scale_step_size
                    ),
                    T.RandomCrop(
                        crop_h,
                        crop_w,
                        pad_value,
                        ignore,
                        random_pad=self.is_train
                    ),
                    T.RandomHorizontalFlip(flip_prob),
                    T.ToTensor(),
                    T.Normalize(
                        mean,
                        std
                    )
                ]
            )
        elif split == 'test':
            self.transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(
                        mean,
                        std
                    )
                ]
            )
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # TODO: handle transform properly when there is no label
        dataset_dict = {}
        assert os.path.exists(self.img_list[index]), 'Path does not exist: {}'.format(self.img_list[index])

        image = self.read_image(self.img_list[index], 'RGB')
        if not self.is_train:
            # Do not save this during training.
            dataset_dict['raw_image'] = image.copy()
        if self.ann_list is not None:
            assert os.path.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.ann_list[index])
            label = self.read_label(self.ann_list[index], self.label_dtype)

            raw_label = label.copy()
            if self.raw_label_transform is not None:
                raw_label = self.raw_label_transform(raw_label, self.ins_list[index])['semantic']
            if not self.is_train:
                # Do not save this during training
                dataset_dict['raw_label'] = raw_label
            
        else:
            label = None

        size = image.shape
        dataset_dict['raw_size'] = np.array(size)
        # To save prediction for official evaluation.
#        name = os.path.splitext(os.path.basename(self.ann_list[index]))[0]
        # TODO: how to return the filename?
        # dataset_dict['name'] = np.array(name)

        # Resize and pad image to the same size before data augmentation.
        if self.split == 'test':
            raw_shape = dataset_dict['raw_image'].shape[:2]
            raw_h = raw_shape[0]
            raw_w = raw_shape[1]
            new_h = (raw_h + 31) // 32 * 32 + 1
            new_w = (raw_w + 31) // 32 * 32 + 1
            image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            image[:, :] = MEAN
            image[:raw_h, :raw_w, :] = dataset_dict['raw_image']
            
            # image, label = self.pre_augmentation_transform(image, label)
            size = image.shape
            dataset_dict['size'] = np.array(size)
        else:
            dataset_dict['size'] = dataset_dict['raw_size']

        # Apply data augmentation.
        image, label = self.transform(image, label)

        dataset_dict['image'] = image
        
        if self.split != 'test':
            # Generate training target.
            label_dict = self.target_transform(label, self.ins_list[index])
            for key in label_dict.keys():
                dataset_dict[key] = label_dict[key]

        return dataset_dict
    
    def _get_files(self, data, dataset_split):
        """Gets files for the specified data type and dataset split.
        Args:
            data: String, desired data ('image' or 'label').
            dataset_split: String, dataset split ('train', 'val', 'test')
        Returns:
            A list of sorted file names or None when getting label for test set.
        """
        if data == 'label' and dataset_split == 'test':
            return None
        pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
        search_files = os.path.join(
            self.root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
        filenames = glob.glob(search_files)
        return sorted(filenames)

    def train_id_to_eval_id(self):
        return _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID
    
    def rgb2id(self, color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def create_label_colormap(self):
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap
        

    def read_image(self, file_name, format=None):
        image = Image.open(file_name)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image

    def read_label(self, file_name, dtype='uint8'):
        # In some cases, `uint8` is not enough for label
        label = Image.open(file_name)
        return np.asarray(label, dtype=dtype)

    def reverse_transform(self, image_tensor):
        """Reverse the normalization on image.
        Args:
            image_tensor: torch.Tensor, the normalized image tensor.
        Returns:
            image: numpy.array, the original image before normalization.
        """
        dtype = image_tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image_tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image_tensor.device)
        image_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
        image = image_tensor.mul(255)\
                            .clamp(0, 255)\
                            .byte()\
                            .permute(1, 2, 0)\
                            .cpu().numpy()
        return image