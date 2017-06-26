import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms


class BinaryClassificationTask():
    def __init__(self, args):
        self.classes = list(range(10))
        self.args = args

        trans = transforms.Compose([transforms.ToTensor()])
        self.train_set = dset.MNIST(root='./data/', train=True, transform=trans, download=args.download)
        self.test_set = dset.MNIST(root='./data/', train=False, transform=trans)
        
    def salt_and_pepper_noise(self, x):
        probs = torch.rand(*x.size())
        x[probs < self.args.noise_prob / 2] = 0
        x[probs > 1 - self.args.noise_prob / 2] = 1
        return x
        
    def init(self):
        labels = np.random.choice(self.classes, 2, replace=False)
        
        # For the next task we won't encounter the same label again
        self.classes = [x for x in self.classes if x not in labels] 
        
        print('Binary classification between {} and {}'.format(labels[0], labels[1]))
        
        train_set = self.convert2tensor(self.train_set, labels, train=True)
        test_set = self.convert2tensor(self.test_set, labels)
        
        train_loader = DataLoader(train_set, 
                                  self.args.batch_size, 
                                  shuffle=True)
        
        test_loader = DataLoader(test_set, 
                                 self.args.batch_size, 
                                 shuffle=True)
        
        return train_loader, test_loader

    def convert2tensor(self, dset, labels, train=False):
        x_set = []
        y_set = []
        
        for x, y in dset:
            if y == labels[0]:
                x_set.append(x)
                y_set.append(torch.LongTensor([0]))
            elif y == labels[1]:
                x_set.append(x)
                y_set.append(torch.LongTensor([1]))
        
        x_set = torch.cat(x_set, 0)
        x_set = x_set.view(x_set.size()[0], -1)
        if train:
            x_set = self.salt_and_pepper_noise(x_set)
        
        y_set = torch.cat(y_set, 0)
        dataset = TensorDataset(x_set, y_set)
        
        return dataset