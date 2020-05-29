import time

import torch as t
import torch.nn as nn


class BasicModule(nn.Module):
    """
    封装nn.Module，提供save()和load()方法
    其他model继承BasicModel，拥有了save()和load()方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，以模型+时间为文件名
        """
        if name is None:
            prefix = 'checkpoints/'+self.model_name+'_'
            name = time.strftime(prefix+'%m%d_%H:%M:%s.pth')
        t.save(self.state_dict(), name)
        return name





