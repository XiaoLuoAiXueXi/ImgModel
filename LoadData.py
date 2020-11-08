import torch
import torchvision
from torchvision.transforms import transforms

data_transform = transforms.Compose([
    #对图像大小统一
    transforms.CenterCrop(224),
    #图像翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

])


class LoadData():
    def __init__(self,batch_size,path):
        self.pathTrain,self.pathTest=path
        # 这个遍历是可以index的[][]
        self.train_dataset = torchvision.datasets.ImageFolder(root=self.pathTrain,transform=data_transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)

        # test一次检测一个
        self.val_dataset = torchvision.datasets.ImageFolder(root=self.pathTest, transform=data_transform)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size = 1, shuffle=True, num_workers=0)
    def getTrainDataset(self):
        return self.train_loader

    def getTestDataset(self):
        return  self.val_loader
