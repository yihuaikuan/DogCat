import os
from PIL import Image
from torch.utils import data
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from config import DefaultConfig
from torchvision.transforms import ToPILImage


class DogCat(data.Dataset):
    def __init__(self, root, transform=None, train=True, test=False):
        """
        获取所有图片地址，根据训练、验证、测试划分数据
        继承DataSet要实现 __getitem()__和 __len()__
        """
        self.test = test
        # 存储所有图片的路径
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # 训练集图片格式： data/train/dog.2131.jpg
        # 测试集图片格式： data/test/1223.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: x.split('.')[-2].split('/'[-1]))  # 测试集样本编号
        else:
            imgs = sorted(imgs, key=lambda x: x.split('.')[-2])

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if not transform:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            # 测试集和验证集
            if self.test or not train:
                self.transfoms = transforms.Compose([transforms.Resize(224),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     normalize])
            else:
                self.transfoms = transforms.Compose([transforms.Resize(256),
                                                     transforms.RandomResizedCrop(224),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     normalize])

    def __getitem__(self, index):
        """
        返回图片，若是测试集则无label
        """
        img_path = self.imgs[index]
        if self.test:
            label = img_path.split('.')[-2].split('/'[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0

        # 将文件读取等费时操作放在__getitem()__中，利用多进程加速
        data = Image.open(img_path)
        data = self.transfoms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    opt = DefaultConfig()
    train_set = DogCat(opt.train_data_root, train=True)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers)
    x, y = train_set[29]
    print(x.shape)

    for d in train_loader:
        # dataloader每次返回一个列表，长度为2，第一个元素为一个batch的数据tensor，第二个为标签tesnor
        # d的类型： <class 'list'>, d的大小：2
        # d[0]的类型：<class 'torch.Tensor'>, d的大小：torch.Size([128, 3, 224, 224])
        # d[1]的类型：<class 'torch.Tensor'>, d的大小：torch.Size([128])
        print('d的类型：{}, d的大小：{}'.format(type(d), len(d)))
        print('d[0]的类型：{}, d的大小：{}'.format(type(d[0]), d[0].shape))
        print('d[1]的类型：{}, d的大小：{}'.format(type(d[1]), d[1].shape))

        break
    data, label = train_set[23]
    showImage = ToPILImage()
    showImage(data).show()
