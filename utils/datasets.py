import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random,uniform
from os.path import join, dirname
import bisect
import warnings
import os
import h5py
from torchvision.utils import save_image
from PIL import ImageFilter


class Class_Balance_PACS_Dataset(Dataset):
    def __init__(self, file_paths, n_class, n_domain,img_transformer=None):
        self.images = []
        self.labels = []
        self.domains = []
        self.n_class = n_class
        count = 0
        for file_path in file_paths:
            f = h5py.File(file_path, "r")
            self.images.append(np.array(f['images']))
            self.labels.append(np.array(f['labels']))
            domain = np.ones(self.images[count].shape[0]) * count
            count += 1
            self.domains.append(domain.copy())
            f.close()
        self.images = np.concatenate(self.images,axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.domains = np.concatenate(self.domains, axis=0)

        self._image_transformer = img_transformer
        self.class_flag = np.zeros(n_class)
        self.domain_flag = np.zeros(n_domain)

    def __getitem__(self, index):
        img = self.images[index]
        label = int(self.labels[index] - 1)
        domain = int(self.domains[index])
        while self.class_flag[label] == 1: # or self.domain_flag[domain] == 1
            index = self.rand_another(index)
            label = int(self.labels[index] - 1)
            img = self.images[index]
            domain = int(self.domains[index])
        self.class_flag[label] = 1
        if np.sum(self.class_flag) == self.n_class:
            self.class_flag = np.zeros(self.n_class)


        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = self._image_transformer(img)
        # save_image(img,'out.jpg')

        # img = transforms.ToTensor()(img)
        return img, label, domain

    def __len__(self):
        return self.images.shape[0]

    def rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.labels != self.labels[idx])[0]
        return np.random.choice(pool)

class Domain_Balance_PACS_Dataset(Dataset):
    def __init__(self, file_paths, n_class, n_domain, img_transformer=None):
        self.images = []
        self.labels = []
        self.domains = []
        self.n_class = n_class
        self.n_domain = n_domain
        count = 0
        for file_path in file_paths:
            f = h5py.File(file_path, "r")
            self.images.append(np.array(f['images']))
            self.labels.append(np.array(f['labels']))
            domain = np.ones(self.images[count].shape[0]) * count
            count += 1
            self.domains.append(domain.copy())
            f.close()
        self.images = np.concatenate(self.images,axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.domains = np.concatenate(self.domains, axis=0)

        self._image_transformer = img_transformer
        self.class_flag = np.zeros(n_class)
        self.domain_flag = np.zeros(n_domain)

    def __getitem__(self, index):
        img = self.images[index]
        label = int(self.labels[index] - 1)
        domain = int(self.domains[index])
        while self.domain_flag[domain] == 1: # or self.domain_flag[domain] == 1
            index = self.rand_another(index)
            label = int(self.labels[index] - 1)
            img = self.images[index]
            domain = int(self.domains[index])
        self.class_flag[label] = 1
        if np.sum(self.domain_flag) == self.n_domain:
            self.domain_flag = np.zeros(self.n_domain)


        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = self._image_transformer(img)
        # save_image(img,'out.jpg')

        # img = transforms.ToTensor()(img)
        return img, label, domain

    def __len__(self):
        return self.images.shape[0]

    def rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.domains != self.domains[idx])[0]
        return np.random.choice(pool)






class PACS_Dataset(Dataset):
    def __init__(self, file_paths, img_transformer=None):
        self.images = []
        self.labels = []
        self.domains = []
        count = 0
        for file_path in file_paths:
            f = h5py.File(file_path, "r")
            self.images.append(np.array(f['images']))
            self.labels.append(np.array(f['labels']))
            domain = np.ones(self.images[count].shape[0]) * count
            count += 1
            self.domains.append(domain.copy())
            f.close()
        self.images = np.concatenate(self.images,axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.domains = np.concatenate(self.domains, axis=0)

        self._image_transformer = img_transformer

    def __getitem__(self, index):
        img = self.images[index]
        label = int(self.labels[index] - 1)
        domain = int(self.domains[index])

        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = self._image_transformer(img)
        # save_image(img,'out.jpg')

        # img = transforms.ToTensor()(img)
        return img, label, domain

    def __len__(self):
        return self.images.shape[0]

class Contrastive_PACS_Dataset(Dataset):
    def __init__(self, file_paths,  img_transformer=None, jig = False, grid_size = 3,jig_classes = 30, bias_whole_image = 0.2):
        self.images = []
        self.labels = []
        self.domains = []
        count = 0
        for file_path in file_paths:
            f = h5py.File(file_path, "r")
            self.images.append(np.array(f['images']))
            self.labels.append(np.array(f['labels']))
            domain = np.ones(self.images[count].shape[0]) * count
            count += 1
            self.domains.append(domain.copy())
            f.close()
        self.images = np.concatenate(self.images,axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.domains = np.concatenate(self.domains, axis=0)

        self._image_transformer = img_transformer
        self._final_transformer =  transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.jig = jig
        if jig == True:
            self.jigsaw_generator = jigsaw_generator(grid_size = grid_size, jig_classes = jig_classes, bias_whole_image = bias_whole_image)

    def __getitem__(self, index):
        img = self.images[index]
        label = int(self.labels[index] - 1)
        domain = int(self.domains[index])

        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = self._image_transformer(img)
        img2 = self._image_transformer(img)
        if self.jig == True:
            jig_img, order = self.jigsaw_generator.transform(img)
        img = self._final_transformer(img)
        img2 = self._final_transformer(img2)

        # save_image(img,'out.jpg')

        # img = transforms.ToTensor()(img)
        if self.jig == True:
            return jig_img, img2, label, domain
        else:
            return img, img2, label, domain

    def __len__(self):
        return self.images.shape[0]

class jigsaw_generator():
    def __init__(self,grid_size, jig_classes, bias_whole_image = 0.7):
        self.grid_size = grid_size
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.bias_whole_image = bias_whole_image
        self._augment_tile = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self._augment_tile = transforms.Compose(self._augment_tile)
        return

    def transform(self,pil_img):
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(pil_img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        data = self.make_grid(data)
        return data, order

    def make_grid(self, x):
        return torchvision.utils.make_grid(x, self.grid_size, padding=0)

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm



def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(int(args.image_size), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # img_tr = img_tr + [transforms.ToTensor()]

    return transforms.Compose(img_tr)

def get_train_transformers_for_contrative(args):
    img_tr = [transforms.RandomResizedCrop(int(args.image_size), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    # img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # img_tr = img_tr + [transforms.ToTensor()]

    return transforms.Compose(img_tr)

def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
    return transforms.Compose(img_tr)
