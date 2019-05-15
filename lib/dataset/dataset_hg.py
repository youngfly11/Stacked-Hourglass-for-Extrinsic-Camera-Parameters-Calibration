import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import json
import imageio
import os.path as osp
import torch
from lib.dataset.low_level_cues import line_detection

class VanishPointsDataSet(Dataset):
    def __init__(self, data_dir, label_set, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        with open(osp.join('./data/processed/labels', label_set), 'r') as read_f:
            label_file = read_f.readlines()
        self.label_file = label_file
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        line_data = self.label_file[index].split(', ')
        image_file = line_data[0].split('_')
        img = imageio.imread(osp.join(self.data_dir, '{}/{}.jpg'.format(image_file[0], image_file[1])))
        h, w, c = img.shape

        edge_det, line_det = line_detection(img)

        gt_heat = np.zeros((int(h/4), int(w/4)))
        pt = np.array([float(line_data[3])/4, float(line_data[4])/4]).astype(np.int32)
        gt_heat = draw_labelmap(img=gt_heat, sigma=1, pt=pt, type='Gaussian')
        sample = {'image': img, "label": gt_heat, 'label_coord': pt*4, 'line_det': line_det, 'edge_det': edge_det}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.label_file)

class RandomHorizonFlip(object):

    def __call__(self, sample):

        image, label, label_coord, line_det, edge_det =\
            sample['image'], sample['label'], sample['label_coord'], sample['line_det'], sample['edge_det']
        rand_seed = np.random.random()
        if rand_seed > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
            line_det = np.fliplr(line_det).copy()
            edge_det = np.fliplr(edge_det).copy()

        label_coord[0] = image.shape[1]-label_coord[0]
        return {'image': image, 'label': label, 'label_coord': label_coord, 'line_det': line_det, 'edge_det': edge_det}


class NormalizedImage(object):
    mean_bgr = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_bgr = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, sample):
        image, label, label_coord, line_det, edge_det = \
            sample['image'], sample['label'], sample['label_coord'], sample['line_det'], sample['edge_det']
        image = image[:,:, [2,1,0]]
        image = image / 255.0
        image -= self.mean_bgr
        image /= self.std_bgr
        line_det = line_det / 255.0
        edge_det = edge_det / 255.0
        return {'image': image, 'label': label, 'label_coord': label_coord, 'line_det': line_det, 'edge_det': edge_det}


class Numpy2Tensor(object):
    """converting the numpy format to tensor """

    def __call__(self, sample):
        image, label, label_coord, line_det, edge_det = \
            sample['image'], sample['label'], sample['label_coord'], sample['line_det'], sample['edge_det']
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        line_det = torch.FloatTensor(line_det)
        edge_det = torch.FloatTensor(edge_det)
        label_coord = torch.FloatTensor(label_coord)
        return {'image': image, 'label': label, 'label_coord': label_coord, 'line_det': line_det, 'edge_det': edge_det}


def get_image_dataloader(batch_size_train=None, batch_size_val=None, batch_size_test=None):

    data_dir = '/root/project/SLAM/SLAM-Extrinsic-Estimation/data/processed/VP_Img_resize'

    data_transforms = {
        'train': transforms.Compose([
            RandomHorizonFlip(),
            NormalizedImage(),
            Numpy2Tensor()
        ]),
        'val': transforms.Compose([
            NormalizedImage(),
            Numpy2Tensor()
        ]),
        'test': transforms.Compose([
            NormalizedImage(),
            Numpy2Tensor()
        ])
    }

    training_dataset = VanishPointsDataSet(data_dir=data_dir, label_set='training.txt', transform=data_transforms['train'])
    val_dataset = VanishPointsDataSet(data_dir=data_dir, label_set='val.txt', transform=data_transforms['val'])
    test_dataset = VanishPointsDataSet(data_dir=data_dir, label_set='test.txt', transform=data_transforms['test'])

    training_loader = DataLoader(dataset=training_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=16, pin_memory=True)

    return training_loader, val_loader, test_loader


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


if __name__ == '__main__':

    train_loader, val_loader, test_loader = get_image_dataloader(batch_size_train=16, batch_size_val=16, batch_size_test=16)

    for batch_idx, data_sample in enumerate(test_loader):

        print(batch_idx, data_sample['image'].shape, data_sample['label'].shape)



