import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import  numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from Vgg16 import Vgg16
from LoadData import LoadData


'''
函数部分
'''
def Train(epochs):

    # ?这个用来干啥
    train_loss=[]
    # 第一层是迭代的次数
    for epoch in range(epochs):
        # 一个epoch跑完所有的图片一次，所以这里是跑完2000张需要的batch次数
        for i,data in enumerate(train):
            '''
            inputs.shape(batch_size,(c,h,w))
            labels.shape(batch_size,1)
            '''
            train_batch_loss=[]
            inputs,labels=data
            # 将这些数据转换成Variable类型
            inputs, labels = Variable(inputs).cuda(1), Variable(labels).cuda(1)
            # 接下来就是跑模型的环节了，我们这里使用print来代替
            # print("epoch：", epoch, "的第" , i, "个inputs") #, inputs.data.size(), "labels", labels.data.size())

            out=net(inputs)     # .cpu().detach().data()
            loss=loss_func(out,labels)
            # 清空上一步的残余更新参数值
            # 这几步都要研究一下
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())
            print("epoch：", epoch, "的第" , i, "个inputs",epoch, i*batch_size, loss.item())
            torch.cuda.empty_cache()
        loss_epoch=np.sum(train_batch_loss)/len(train)
        scheduler.step(loss_epoch)
        train_loss.append(loss_epoch)
    return train_loss

def Test():
    total_correct = 0
    total=0
    for i,data in enumerate(test):
        x,y =data
        total=total+1
        '''
        !!!检查一下输入的维度是多少，之前是(n,c,h,w)
        
        '''
        x = x.cuda(1)
        out = net(x)
        out = out.cpu().detach().numpy()
        #out输出是二维向量，找到大的就是概率大的
        pred=np.argmax(out)
        if pred==y:
            total_correct += 1
            print(i,end='')

    acc = total_correct / total
    print('\ntest acc:', acc)

    torch.cuda.empty_cache()

def Draw(epoch,*loss):
    '''
    这里不知道为什么list变成了numpy了，np.array(list).reshape()改变他的形状
    可以
    :param epoch: 次数
    :param loss: 一个list,不知道为什么变成了了一个list
    :return:
    '''
    y=np.array(loss).reshape(epoch,)
    x=epoch
    plt.plot(range(x),y,'-r')
    plt.savefig('output.png')



'''
调用函数，读取数据集
训练网络
'''

print(torch.cuda.is_available())
device=torch.device('cuda:1')
# 数据集路径
batch_size=32
train=LoadData(batch_size).getTrainDataset()
test=LoadData(8).getTestDataset()
net=Vgg16().cuda(1)
loss_func =nn.CrossEntropyLoss()
# 需要把参数传递进去进行优化
# 这个不要放到循环离去，否则会循环定义，显存会爆炸
optimizer=optim.Adam(net.parameters(),lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
#!!!需要探索一些多GPU运行机理
# net = torch.nn.DataParallel(Model.cuda(), device_ids=[3])
epochs=100
loss=Train(epochs)
print(type(loss))
print(loss)
Draw(epochs,loss)
Test()