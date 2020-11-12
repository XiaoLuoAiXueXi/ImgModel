import torch
import torchvision
from torch import  nn
from torch import optim

class Vgg16(nn.Module):
    def __init__(self,numclass=2):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            # 没在结构里看到Batch,不太懂这个num_feature的作用
            nn.BatchNorm2d(num_features=64,eps=1e-5,momentum=0.1,affine=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256,eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv5的输出为7*7*512
        # ！！！！这里好像少一个view
        self.dense=nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,numclass)
        )
        # 输出为两类，所以为（4096,2）


    def forward(self,x):
        # 单独的层可以作为函数，合在一起的几个层也能作为一个函数
        # 为什么单独的层能看做函数呢
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        # 加上一行这个，
        x=x.view(-1,7*7*512)

        x=self.dense(x)
        return x


