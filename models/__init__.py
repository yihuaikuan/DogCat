from .AlexNet import AlexNet
from .MyResNet import ResNet

# 这样可以在主函数中写
from models import AlexNet
# 或
# import models
# model = models.AlexNet
# 或
# import models
# model = getattr('models', 'AlexNet')
