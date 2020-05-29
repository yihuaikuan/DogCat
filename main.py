from torch.utils.data import DataLoader
import train

from data.dataset import DogCat
from config import DefaultConfig
from torchvision import models

model = models.resnet34(num_classes=2)


opt = DefaultConfig()
train_set = DogCat(opt.train_data_root, train=True)
train_loader = DataLoader(dataset=train_set,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          num_workers=opt.num_workers)

train.fit(model, train_loader, 1000)
