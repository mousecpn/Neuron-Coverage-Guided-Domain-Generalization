from torch.utils.data import DataLoader
import torch
import argparse
from torch import nn
from torch.autograd import Variable
from utils.datasets import *
from torch.nn import functional as F
from torch import optim
import os
from torchvision.utils import save_image
from torch.distributions import Beta
from resnet import *
from utils.optim import *
import math


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
# pacs2index = {"art_painting":0, "cartoon":1, "photo":2, "sketch":3}
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")

    parser.add_argument("--learning_rate", "-l", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()

class Trainer():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        os.makedirs('./checkpoints', exist_ok=True)
        # r18
        model = resnet18(pretrained=False, num_classes=args.n_classes)
        weight = torch.load("/home/dailh/.cache/torch/checkpoints/resnet18-5c106cde.pth")
        weight['fc.weight'] = model.state_dict()['fc.weight']
        weight['fc.bias'] = model.state_dict()['fc.bias']
        model.load_state_dict(weight, strict=False)
        self.model = model.to(device)
        # print(self.model)

        # self.adv = nn.Sequential(nn.Linear(512, 3)).to(device)



        train_data = ['art_painting_train.hdf5',
                      'cartoon_train.hdf5',
                      'photo_train.hdf5',
                      'sketch_train.hdf5']

        val_data = ['art_painting_val.hdf5',
                    'cartoon_val.hdf5',
                    'photo_val.hdf5',
                    'sketch_val.hdf5']

        test_data = ['art_painting_test.hdf5',
                     'cartoon_test.hdf5',
                     'photo_test.hdf5',
                     'sketch_test.hdf5']
        data_path = "data/Train val splits and h5py files pre-read"

        train_paths = []
        unseen_index = pacs_datasets.index(args.target)
        for data in train_data:
            path = os.path.join(data_path, data)
            train_paths.append(path)

        val_paths = []
        for data in val_data:
            path = os.path.join(data_path, data)
            val_paths.append(path)

        unseen_data_path = [os.path.join(data_path, test_data[unseen_index])]
        train_paths.remove(train_paths[unseen_index])
        val_paths.remove(val_paths[unseen_index])

        img_tr = get_train_transformers_for_contrative(args)
        train_dataset = Contrastive_PACS_Dataset(train_paths, img_tr, jig = False)
        self.source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                             drop_last=True)
        img_tr = get_val_transformer(args)
        val_dataset = PACS_Dataset(val_paths, img_tr)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                             drop_last=False)
        target_dataset = PACS_Dataset(unseen_data_path, img_tr)
        self.target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                             drop_last=False)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
            len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, nesterov=args.nesterov)

        self.current_epoch = 0
        self.count = 0
        self.n_classes = args.n_classes
        self.criterion = nn.CrossEntropyLoss()
        self.L2norm = nn.MSELoss()

        # NCDG
        self.lambdas = 1
        self.beta = 0.01
        self.t = 0.005

        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        return

    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self._do_epoch()
            adjust_learning_rate(self.optimizer, self.current_epoch, self.args.epochs, self.args.learning_rate)
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))

        return self.model

    def do_test(self, loader):
        class_correct = 0
        for it, (data, class_l, _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct

    def _do_epoch(self):
        self.model.train()
        for it, (data1, data2, class_l, d_idx) in enumerate(self.source_loader):
            data1, data2, class_l, d_idx = data1.to(self.device), data2.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.count += 1
            self.optimizer.zero_grad()

            class_logit1, feats = self.model(data1, return_feature=True)

            class_loss1 = self.criterion(class_logit1, class_l)
            coverage1 = self.coverage(feats)
            conv1 = class_loss1 - coverage1
            grad1 = torch.autograd.grad(conv1, self.model.parameters(), create_graph=True, retain_graph=True, allow_unused=True)

            class_logit2, feats = self.model(data2, return_feature=True)

            class_loss2 = self.criterion(class_logit2, class_l)
            coverage2 = self.coverage(feats)
            conv2 = class_loss2 - coverage2
            grad2 = torch.autograd.grad(conv2, self.model.parameters(), create_graph=True, retain_graph=True,
                                        allow_unused=True)

            loss = self.beta * self.sim(grad1, grad2) + conv1 + conv2

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            if it % 30 == 0:
                print("{}/{} iter/epoch, [losses] class: {}, total: {}. ".format(it, self.current_epoch, class_loss1.item(), loss.item()))

        self.model.eval()
        with torch.no_grad():
            print('epoch:', self.current_epoch)
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.results[phase][self.current_epoch] = class_acc
                print(phase, ':', class_acc)

    def coverage(self, feats):
        loss = 0.0
        for f in feats:
            for i in range(f.shape[0]):
                loss += self.normalize(f[i])
        loss /= f.shape[0]

        return self.lambdas * loss

    def normalize(self, neuron):
        max = torch.max(neuron)
        min = torch.min(neuron)
        out = (neuron - min)/(max - min)
        out[out < self.t] = 0.0
        return out.mean()

    def sim(self, grad1, grad2):
        loss = 0.0
        for i in range(len(grad1)):
            loss += (grad1[i] - grad2[i]).pow(2).sum().sqrt()
        return loss





if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

