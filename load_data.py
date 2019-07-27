# -*- coding: utf-8 -*-
# @Author  : 胡子旋
# @Email   ：1017190168@qq.com

import torch
import os,glob
import visdom
import time
import torchvision
import random,csv
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

class pokemom(Dataset):
    def __init__(self,root,resize,mode,):
        super(pokemom,self).__init__()
        # 保存参数
        self.root=root
        self.resize=resize
        # 给每一个类做映射
        self.name2label={}  # "squirtle":0 ,"pikachu":1……
        for name in sorted(os.listdir(os.path.join(root))):
            # 过滤掉文件夹
            if not os.path.isdir(os.path.join(root,name)):
                continue
            # 保存在表中;将最长的映射作为最新的元素的label的值
            self.name2label[name]=len(self.name2label.keys())
        print(self.name2label)
        # 加载文件
        self.images,self.labels=self.load_csv('images.csv')
        # 裁剪数据
        if mode=='train':
            self.images=self.images[:int(0.6*len(self.images))]   # 将数据集的60%设置为训练数据集合
            self.labels=self.labels[:int(0.6*len(self.labels))]   # label的60%分配给训练数据集合
        elif mode=='val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]  # 从60%-80%的地方
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.images))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]   # 从80%的地方到最末尾
            self.labels = self.labels[int(0.8 * len(self.labels)):]
        # image+label 的路径
    def load_csv(self,filename):
        # 将所有的图片加载进来
        # 如果不存在的话才进行创建
        if not os.path.exists(os.path.join(self.root,filename)):
            images=[]
            for name in self.name2label.keys():
                images+=glob.glob(os.path.join(self.root,name,'*.png'))
                images+=glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images),images)
            # 1167 'pokeman\\bulbasaur\\00000000.png'
            # 将文件以上述的格式保存在csv文件内
            random.shuffle(images)
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                writer=csv.writer(f)
                for img in images:    #  'pokeman\\bulbasaur\\00000000.png'
                    name=img.split(os.sep)[-2]
                    label=self.name2label[name]
                    writer.writerow([img,label])
                print("write into csv into :",filename)

        # 如果存在的话就直接的跳到这个地方
        images,labels=[],[]
        with open(os.path.join(self.root, filename)) as f:
            reader=csv.reader(f)
            for row in reader:
                # 接下来就会得到 'pokeman\\bulbasaur\\00000000.png' 0 的对象
                img,label=row
                # 将label转码为int类型
                label=int(label)
                images.append(img)
                labels.append(label)
        # 保证images和labels的长度是一致的
        assert len(images)==len(labels)
        return images,labels


    # 返回数据的数量
    def __len__(self):
        return len(self.images)   # 返回的是被裁剪之后的关系

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean
        return x
    # 返回idx的数据和当前图片的label
    def __getitem__(self,idx):
        # idex-[0-总长度]
        # retrun images,labels
        # 将图片，label的路径取出来
        # 得到的img是这样的一个类型：'pokeman\\bulbasaur\\00000000.png'
        # 然而label得到的则是 0，1，2 这样的整形的格式
        img,label=self.images[idx],self.labels[idx]
        tf=transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),  # 将t图片的路径转换可以处理图片数据
            # 进行数据加强
            transforms.Resize((int(self.resize*1.25),int(self.resize*1.25))),
            # 随机旋转
            transforms.RandomRotation(15),   # 设置旋转的度数小一些，否则的话会增加网络的学习难度
            # 中心裁剪
            transforms.CenterCrop(self.resize),   # 此时：既旋转了又不至于导致图片变得比较的复杂
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])

        ])
        img=tf(img)
        label=torch.tensor(label)
        return img,label




def main():
    # 验证工作
    viz=visdom.Visdom()

    db=pokemom('pokeman',64,'train')  # 这里可以改变大小 224->64,可以通过visdom进行查看
    # 可视化样本
    x,y=next(iter(db))
    print('sample:',x.shape,y.shape,y)
    viz.image(db.denormalize(x),win='sample_x',opts=dict(title='sample_x'))
    # 加载batch_size的数据
    loader=DataLoader(db,batch_size=32,shuffle=True,num_workers=8)
    for x,y in loader:
        viz.images(db.denormalize(x),nrow=8,win='batch',opts=dict(title='batch'))
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch-y'))
        # 每一次加载后，休息10s
        time.sleep(10)

if __name__ == '__main__':
    main()