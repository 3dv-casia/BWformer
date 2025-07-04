import numpy as np
from datasets.corners import CornersDataset
import os
import skimage
import cv2
from torchvision import transforms
from PIL import Image
from datasets.data_utils import RandomBlur
import torch

class OutdoorBuildingDataset(CornersDataset):
    def __init__(self, data_path, det_path, phase='train', image_size=256, rand_aug=True,
                 inference=False):
        super(OutdoorBuildingDataset, self).__init__(image_size, inference)
        self.data_path = data_path
        self.det_path = det_path
        self.phase = phase
        self.rand_aug = rand_aug
        self.image_size = image_size
        self.inference = inference

        self.train_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0),
            transforms.RandomGrayscale(p=0)])

        if phase == 'train':
            datalistfile = os.path.join(data_path, 'train_list.txt')
            self.training = True
        else:
            datalistfile = os.path.join(data_path, 'train_list.txt')
            self.training = False
        with open(datalistfile, 'r') as f:
            _data_names = f.readlines()
        if phase == 'train':
            self._data_names = _data_names
        else:
            if phase == 'valid':
                self._data_names = _data_names[:50]
            elif phase == 'test':
                self._data_names = _data_names[50:]
            else:
                raise ValueError('Invalid phase {}'.format(phase))

    def __len__(self):
        return len(self._data_names)

    def __getitem__(self, idx):
        data_name = self._data_names[idx][:-1]
        annot_path = os.path.join(self.data_path, 'annot', data_name + '.npy')
        annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
        img_path = os.path.join(self.data_path, 'rgb', data_name + '.jpg')
        rgb = cv2.imread(img_path)
    
        if self.image_size != 256:
            rgb, annot = self.resize_data(rgb, annot)
        corners = np.array(list(annot.keys()))[:, [1, 0]]
        
        if self.rand_aug:
            image,  annot, corner_mapping, _ = self.random_aug_annot(rgb, annot)
        else:
            image = rgb
            
        rec_mat = None
      
        corners = np.array(list(annot.keys()))[:, [1, 0]]


        if not self.inference and len(corners) > 100:
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        if self.training:
            corners += np.random.normal(0, 0, size=corners.shape)
            pil_img = Image.fromarray(image)
            image = self.train_transform(pil_img)
            image = np.array(image)
            
        image = skimage.img_as_float(image)
        
        sort_idx = np.lexsort(corners.T)
        corners = corners[sort_idx]

        corner_list = []
        for corner_i in range(corners.shape[0]):
            corner_list.append((corners[corner_i][1], corners[corner_i][0]))
            
        raw_data = {
            'name': data_name,
            'corners': corner_list,
            'annot': annot,
            'image': image,
            'rec_mat': rec_mat,
            'annot_path': annot_path,
            'img_path': img_path,
        }
        
        return self.process_data(raw_data)

    def random_aug_annot(self, img, annot, det_corners=None):
        img, annot, det_corners = self.random_flip(img, annot, det_corners)
        theta = np.random.randint(0, 360) / 360 * np.pi * 2
        r = self.image_size / 256
        origin = [127 * r, 127 * r]
        p1_new = [127 * r + 100 * np.sin(theta) * r, 127 * r - 100 * np.cos(theta) * r]
        p2_new = [127 * r + 100 * np.cos(theta) * r, 127 * r + 100 * np.sin(theta) * r]
        p1_old = [127 * r, 127 * r - 100 * r]  # y_axis
        p2_old = [127 * r + 100 * r, 127 * r]  # x_axis
        pts1 = np.array([origin, p1_old, p2_old]).astype(np.float32)
        pts2 = np.array([origin, p1_new, p2_new]).astype(np.float32)
        M_rot = cv2.getAffineTransform(pts1, pts2)

        all_corners = list(annot.keys())
        if det_corners is not None:
            for i in range(det_corners.shape[0]):
                all_corners.append(tuple(det_corners[i]))
        all_corners_ = np.array(all_corners)[:, [0, 1]]
        corner_mapping = dict()
        ones = np.ones([all_corners_.shape[0], 1])
        all_corners_ = np.concatenate([all_corners_, ones], axis=-1)
        aug_corners = np.array(all_corners)
        aug_corners[:, [0, 1]] = np.matmul(M_rot, all_corners_.T).T


        for idx, corner in enumerate(all_corners):
            corner_mapping[corner] = aug_corners[idx]

        new_corners = np.array(list(corner_mapping.values()))
        if new_corners.min() <= 0 or new_corners.max() >= (self.image_size - 1):
            return img, annot, None, det_corners

        aug_annot = dict()
        for corner, connections in annot.items():
            new_corner = corner_mapping[corner]
            tuple_new_corner = tuple(new_corner)
            aug_annot[tuple_new_corner] = list()
            for to_corner in connections:
                aug_annot[tuple_new_corner].append(corner_mapping[tuple(to_corner)])

        rows, cols, ch = img.shape
        new_img = cv2.warpAffine(img, M_rot, (cols, rows), borderValue=(255, 255, 255))
        
        y_start = (new_img.shape[0] - self.image_size) // 2
        x_start = (new_img.shape[1] - self.image_size) // 2
        aug_img = new_img[y_start:y_start + self.image_size, x_start:x_start + self.image_size, :]
        
        if det_corners is None:
            return aug_img, aug_annot, corner_mapping, None
        else:
            aug_det_corners = list()
            for corner in det_corners:
                new_corner = corner_mapping[tuple(corner)]
                aug_det_corners.append(new_corner)
            aug_det_corners = np.array(aug_det_corners)
            return aug_img,  aug_annot, corner_mapping, aug_det_corners


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATAPATH = './data/cities_dataset'
    DET_PATH = './data/det_final'
    train_dataset = OutdoorBuildingDataset(DATAPATH, DET_PATH, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                                  collate_fn=collate_fn)
    for i, item in enumerate(train_dataloader):
        import pdb;

        pdb.set_trace()
        print(item)
