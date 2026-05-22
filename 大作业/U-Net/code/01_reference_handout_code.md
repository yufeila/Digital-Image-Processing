
## 1. 数据下载/路径配置

```
%env no_proxy='a.test.com,127.0.0.1,2.2.2.2'

!wget https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/20230927/Unet.zip

!unzip Unet.zip
```

安装ml_collections(以下命令再终端中运行）

```
pip install ml_collections -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```

导入实验环境

```
#导入实验所需要的库

import os

import math

import numpy as np

from PIL import Image, ImageSequence

import matplotlib.pyplot as plt

import mindspore

device_id = int(0)

mindspore.set_context(device_target="Ascend", save_graphs=False)

mindspore.set_seed(1)

```

## 2. 数据读取与预处理

步骤 1 展示数据集图像和标签

```
#显示下载好的数据

train_image_path = "Unet/data/train-volume.tif"

train_masks_path = "Unet/data/train-labels.tif

image = np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(train_image_path))])

masks = np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(train_masks_path))])

#展示图片

def show_image(image_list,num = 6):

    img_titles = []

    img_draws = []

    for ind,img in enumerate(image_list):

        if ind == num:

            break

        img_titles.append(ind)

        img_draws.append(img)

    for i in range(len(img_titles)):

        if len(img_titles) > 6:

            row = 3

        elif 3<len(img_titles)<=6:

            row = 2

        else:

            row = 1

        col = math.ceil(len(img_titles)/row)

        plt.subplot(row,col,i+1),plt.imshow(img_draws[i],'gray')

        plt.title(img_titles[i])

        plt.xticks([]),plt.yticks([])

    plt.show()

show_image(image,num = 12)

show_image(masks,num = 12)
```

步骤 5 数据增强

```
import cv2

import os

import random

import mindspore.dataset as ds

import glob

import mindspore.dataset.vision as vision_C  

import mindspore.dataset.transforms as C_transforms

from mindspore.dataset.vision import Inter

def train_transforms(img_size):

    return [

    vision_C.Resize(img_size, interpolation=Inter.NEAREST),

    vision_C.Rescale(1./255., 0.0),          # 将像素值缩放到范围 [0, 1]

    vision_C.RandomHorizontalFlip(prob=0.5), # 以 0.5 的概率进行随机水平翻转

    vision_C.RandomVerticalFlip(prob=0.5),   # 以 0.5 的概率进行随机垂直翻转

    vision_C.HWC2CHW()                       # 将图像的通道维度从 "HWC"（高度、宽度、通道）顺序转换为 "CHW"（通道、高度、宽度）顺序

    ]

def val_transforms(img_size):

    return [

    vision_C.Resize(img_size, interpolation=Inter.NEAREST),

    vision_C.Rescale(1/255., 0),# 将像素值缩放到范围 [0, 1]

    vision_C.HWC2CHW()          # 将图像的通道维度从 "HWC"（高度、宽度、通道）顺序转换为 "CHW"（通道、高度、宽度）顺序

    ]

class Data_Loader:

    def __init__(self, data_path):

        # 初始化函数，读取所有data_path下的图片

        self.data_path = data_path

        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png')) # 获取所有图像文件的路径

        self.label_path = glob.glob(os.path.join(data_path, 'mask/*.png')) # 获取所有标签文件的路径

    def __getitem__(self, index):

        # 根据index读取图片

        image = cv2.imread(self.imgs_path[index])                           # 读取图像文件

        label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)    # 以灰度模式读取标签文件

        label = label.reshape((label.shape[0], label.shape[1], 1))          # 将标签的通道维度从 2D 转换为 3D

        return image, label

    @property

    def column_names(self):

        # 返回列名

        column_names = ['image', 'label']

        return column_names

    def __len__(self):

        # 返回训练集大小
        return len(self.imgs_path)

def create_dataset(data_dir, img_size, batch_size, augment, shuffle):

    # 创建 Data_Loader 对象

    mc_dataset = Data_Loader(data_path=data_dir)

    # 创建 GeneratorDataset 对象

    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)

    # 根据是否进行数据增强选择转换操作

    if augment:

        transform_img = train_transforms(img_size)

    else:

        transform_img = val_transforms(img_size)

    # 设置随机种子

    seed = random.randint(1,1000)

    mindspore.set_seed(seed)

    # 对标签进行转换操作

    dataset = dataset.map(input_columns='image', num_parallel_workers=1, operations=transform_img)

    mindspore.set_seed(seed)

    # 对图像进行转换操作

    dataset = dataset.map(input_columns="label", num_parallel_workers=1, operations=transform_img)

    # 如果需要进行打乱，将数据集打乱

    if shuffle:

        dataset = dataset.shuffle(buffer_size=10000)

    # 将数据集按批次划分

    dataset = dataset.batch(batch_size, num_parallel_workers=1)

    # 打印数据集大小信息

    if augment == True and shuffle == True:

        print("训练集数据量：", len(mc_dataset))

    elif augment == False and shuffle == False:

        print("验证集数据量：", len(mc_dataset))

    else:

        pass

return dataset
```

调用：

```
if __name__ == '__main__':

     # 创建验证集的数据集

    train_dataset = create_dataset(' ./Unet/ISBI/val', img_size=224, batch_size=3, augment=False, shuffle=False)

    # 遍历数据集并打印图像和标签信息

    for item, (image, label) in enumerate(train_dataset):

        if item < 5:

            # 打印每个批次的图像和标签的形状和数据类型信息

            print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}",'---',f"Shape of label [N, C, H, W]: {label.shape} {label.dtype}")
```



## 3. 搭建U-Net 网络结构

```
import mindspore.nn as nn

import mindspore.numpy as np

import mindspore.ops as ops

from mindspore import Tensor

# 创建包含两个卷积层和归一化层的序列模块

def double_conv(in_ch, out_ch):

    return nn.SequentialCell(nn.Conv2d(in_ch, out_ch, 3),        # 第一个卷积层，输入通道数为 in_ch，输出通道数为 out_ch，卷积核大小为 3x3

                              nn.BatchNorm2d(out_ch), nn.ReLU(), # 归一化层，对输出通道进行归一化；使用ReLU 激活函数

                              nn.Conv2d(out_ch, out_ch, 3),      # 第二个卷积层，输入通道数为 out_ch，输出通道数为 out_ch，卷积核大小为 3x3

                              nn.BatchNorm2d(out_ch), nn.ReLU()) # 归一化层，对输出通道进行归一化；使用ReLU 激活函数

class UNet(nn.Cell):

    def __init__(self, in_ch = 3, n_classes = 1):

        super(UNet, self).__init__()

        # 定义 U-Net 模型的各个组件

        self.double_conv1 = double_conv(in_ch, 64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double_conv2 = double_conv(64, 128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double_conv3 = double_conv(128, 256)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double_conv4 = double_conv(256, 512)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double_conv5 = double_conv(512, 1024)

        self.upsample1 = nn.ResizeBilinear()

        self.double_conv6 = double_conv(1024 + 512, 512)

        self.upsample2 = nn.ResizeBilinear()

        self.double_conv7 = double_conv(512 + 256, 256)

        self.upsample3 = nn.ResizeBilinear()

        self.double_conv8 = double_conv(256 + 128, 128)

        self.upsample4 = nn.ResizeBilinear()

        self.double_conv9 = double_conv(128 + 64, 64)

        self.final = nn.Conv2d(64, n_classes, 1)

        self.sigmoid = ops.sigmoid

    def construct(self, x):

        # U-Net 模型的前向传播逻辑

        feature1 = self.double_conv1(x)

        tmp = self.maxpool1(feature1)

        feature2 = self.double_conv2(tmp)

        tmp = self.maxpool2(feature2)

        feature3 = self.double_conv3(tmp)

        tmp = self.maxpool3(feature3)

        feature4 = self.double_conv4(tmp)

        tmp = self.maxpool4(feature4)

        feature5 = self.double_conv5(tmp)

        up_feature1 = self.upsample1(feature5, scale_factor=2)

        tmp = ops.concat((feature4, up_feature1),axis=1)      

        tmp = self.double_conv6(tmp)

        up_feature2 = self.upsample2(tmp, scale_factor=2)

        tmp = ops.concat((feature3, up_feature2),axis=1)       

        tmp = self.double_conv7(tmp)

        up_feature3 = self.upsample3(tmp, scale_factor=2)

        tmp = ops.concat((feature2, up_feature3),axis=1)        

        tmp = self.double_conv8(tmp)

        up_feature4 = self.upsample4(tmp, scale_factor=2)

        tmp = ops.concat((feature1, up_feature4),axis=1)       

        tmp = self.double_conv9(tmp)

        output = self.sigmoid(self.final(tmp))

        return output
```

自定义评估指标

```
import numpy as np

from mindspore.train import Metric

from mindspore import Tensor

class metrics_(Metric):

    # 初始化

    def __init__(self, metrics, smooth=1e-5):

        super(metrics_, self).__init__()

        self.metrics = metrics

        self.smooth = smooth

        self.metrics_list = [0. for i in range(len(self.metrics))]

        self._samples_num = 0

        self.clear()

    # 计算准确率指标

    def Acc_metrics(self,y_pred, y):

        tp = np.sum(y_pred.flatten() == y.flatten(), dtype=y_pred.dtype)

        total = len(y_pred.flatten())

        single_acc = float(tp) / float(total)

        return single_acc

    # 计算 IoU (Intersection over Union) 指标

    def IoU_metrics(self,y_pred, y):

        intersection = np.sum(y_pred.flatten() * y.flatten())

        unionset = np.sum(y_pred.flatten() + y.flatten()) - intersection

        single_iou = float(intersection) / float(unionset + self.smooth)

        return single_iou

    # 计算 Dice 系数指标

    def Dice_metrics(self,y_pred, y):

        intersection = np.sum(y_pred.flatten() * y.flatten())

        unionset = np.sum(y_pred.flatten()) + np.sum(y.flatten())

        single_dice = 2*float(intersection) / float(unionset + self.smooth)

        return single_dice

    # 计算敏感性指标

    def Sens_metrics(self,y_pred, y):

        tp = np.sum(y_pred.flatten() * y.flatten())

        actual_positives = np.sum(y.flatten())

        single_sens = float(tp) / float(actual_positives + self.smooth)

        return single_sens

    # 计算特异性指标

    def Spec_metrics(self,y_pred, y):

        true_neg = np.sum((1 - y.flatten()) * (1 - y_pred.flatten()))

        total_neg = np.sum((1 - y.flatten()))

        single_spec = float(true_neg) / float(total_neg + self.smooth)

        return single_spec

    # 清空内部的评估结果

    def clear(self):

        """Clears the internal evaluation result."""

        self.metrics_list = [0. for i in range(len(self.metrics))]

        self._samples_num = 0

    # 更新评估结果

    def update(self, *inputs):

        if len(inputs) != 2:

            raise ValueError("For 'update', it needs 2 inputs (predicted value, true value), ""but got {}.".format(len(inputs)))

        y_pred = Tensor(inputs[0]).asnumpy()  # 将输入的预测值转换为NumPy数组

        # y_pred = np.array(Tensor(inputs[0]))  #

        y_pred[y_pred > 0.5] = float(1)       # 将预测值大于0.5的部分设置为1

        y_pred[y_pred <= 0.5] = float(0)      # 将预测值小于等于0.5的部分设置为0      

        y = Tensor(inputs[1]).asnumpy()       # 将输入的真实值转换为NumPy数组

        # y = np.array(Tensor(inputs[1]))     #

        self._samples_num += y.shape[0]

        if y_pred.shape != y.shape:

            raise ValueError(f"For 'update', predicted value (input[0]) and true value (input[1]) "

                             f"should have same shape, but got predicted value shape: {y_pred.shape}, "

                             f"true value shape: {y.shape}.")

        for i in range(y.shape[0]):

            if "acc" in self.metrics:

                single_acc = self.Acc_metrics(y_pred[i], y[i])

                self.metrics_list[0] += single_acc

            if "iou" in self.metrics:

                single_iou = self.IoU_metrics(y_pred[i], y[i])

                self.metrics_list[1] += single_iou

            if "dice" in self.metrics:

                single_dice = self.Dice_metrics(y_pred[i], y[i])

                self.metrics_list[2] += single_dice

            if "sens" in self.metrics:

                single_sens = self.Sens_metrics(y_pred[i], y[i])

                self.metrics_list[3] += single_sens

            if "spec" in self.metrics:

                single_spec = self.Spec_metrics(y_pred[i], y[i])

                self.metrics_list[4] += single_spec

    # 评估模型性能并返回评估指标结果

    def eval(self):

        if self._samples_num == 0:

            raise RuntimeError("The 'metrics' can not be calculated, because the number of samples is 0, "

                               "please check whether your inputs(predicted value, true value) are empty, or has "

                               "called update method before calling eval method.")

        for i in range(len(self.metrics_list)):

            self.metrics_list[i] = self.metrics_list[i] / float(self._samples_num)

        return self.metrics_list
```

调用
``` python
#样本点

x = Tensor(np.array([[[[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.8]]]]))

y = Tensor(np.array([[[[0, 1, 1], [1, 0, 0], [0, 1, 1]]]]))

#实例化了metrics_类，传入评估指标列表["acc", "iou", "dice", "sens", "spec"]和平滑参数smooth=1e-5

metric = metrics_(["acc", "iou", "dice", "sens", "spec"],smooth=1e-5)

#调用clear方法清除之前的评估结果

metric.clear()

#更新评估指标

metric.update(x, y)

#返回最终的评估结果

res = metric.eval()

print( '丨acc: %.4f丨丨iou: %.4f丨丨dice: %.4f丨丨sens: %.4f丨丨spec: %.4f丨' % (res[0], res[1], res[2], res[3],res[4]), flush=True)
```

## 4. loss、optimizer、metric 以及训练循环

```
import mindspore as ms

from mindspore import jit

import ml_collections

from mindspore import load_checkpoint

def get_config():

    # 定义模型参数

    config = ml_collections.ConfigDict()

    config.epochs = 10  # 训练的轮数

    config.train_data_path = "./Unet/ISBI/train/"  # 训练数据集路径F

    config.val_data_path = "ISBI/val/" # 验证数据集路径

    config.imgsize = 224  # 图像尺寸

    config.batch_size = 4  # 批大小

    config.pretrained_path = None  # 预训练模型路径

    config.in_channel = 3  # 输入通道数

    config.n_classes = 1  # 类别数

    config.lr = 0.0001  # 学习率

    return config

cfg = get_config()

#获取训练集和验证集

train_dataset = create_dataset(cfg.train_data_path, img_size=cfg.imgsize, batch_size= cfg.batch_size, augment=True, shuffle = True)

val_dataset = create_dataset(cfg.val_data_path, img_size=cfg.imgsize, batch_size= cfg.batch_size, augment=False, shuffle = False)

def train(model, dataset, loss_fn, optimizer, met):

    # 定义前向传播函数

    def forward_fn(data, label):

        logits = model(data)

        loss = loss_fn(logits, label)

        return loss, logits

    # 计算梯度

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # 定义一步训练

    @jit

    def train_step(data, label):

        (loss, logits), grads = grad_fn(data, label)

        loss = ops.depend(loss, optimizer(grads))

        return loss, logits

    size = dataset.get_dataset_size()   # 获取数据集的大小（样本数量）

    model.set_train(True)               # 设置模型为训练模式

    train_loss = 0                      # 训练损失的累加和

    train_pred = []                     # 存储训练预测结果

    train_label = []                    # 存储训练标签

    # 遍历数据集的每个批次

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):

        loss, logits = train_step(data, label)     # 调用训练步骤函数进行模型训练，返回损失和预测结果

        train_loss += loss.asnumpy()               # 将损失值累加到总和中

        train_pred.extend(logits.asnumpy())        # 将预测结果添加到训练预测列表中

        train_label.extend(label.asnumpy())        # 将标签添加到训练标签列表中

    train_loss /= size                             # 计算平均训练损失                

    metric = metrics_(met, smooth=1e-5)            # 创建评估指标对象

    metric.clear()                                 # 清除评估指标的状态

    metric.update(train_pred, train_label)         # 更新评估指标，传入训练预测结果和标签

    res = metric.eval()                            # 计算评估指标的结果

    # 打印训练损失和评估指标的结果

    print(f'Train loss:{train_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))

def val(model, dataset, loss_fn, met):

    size = dataset.get_dataset_size()  # 获取数据集的大小（样本数量）

    model.set_train(False)             # 设置模型为验证模式

    val_loss = 0                       # 验证损失的累加和

    val_pred = []                      # 存储验证预测结果

    val_label = []                     # 存储验证标签

    # 遍历数据集的每个批次

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):

        pred = model(data)                          #使用模型进行预测

        val_loss += loss_fn(pred, label).asnumpy()  # 计算验证损失并累加到总和中

        val_pred.extend(pred.asnumpy())             # 将预测结果添加到验证预测列表中

        val_label.extend(label.asnumpy())           # 将标签添加到验证标签列表中

    val_loss /= size                      # 计算平均验证损失

    metric = metrics_(met, smooth=1e-5)   # 创建评估指标对象

    metric.clear()                        # 清除评估指标的状态

    metric.update(val_pred, val_label)    # 更新评估指标，传入验证预测结果和标签

    res = metric.eval()                   # 计算评估指标的结果

    # 打印验证损失和评估指标的结果

    print(f'Val loss:{val_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))

    checkpoint = res[1]                    # 选择保存检查点的指标，此处为acc（准确率）

    return checkpoint, res[4]              # 返回用于判断是否保存检查点的指标和用于比较模型性能的指标

net = UNet(cfg.in_channel, cfg.n_classes)                                # 创建UNet模型，输入通道数为cfg.in_channel，类别数为cfg.n_classes

criterion = nn.BCEWithLogitsLoss()                                       # 创建二分类交叉熵损失函数

load_checkpoint("./Unet/checkpoint/best_UNet_own.ckpt ", net)  # 加载已有的模型参数

parameters = net.final.get_parameters()                # 只优化最后一层参数

optimizer = nn.SGD(params=parameters, learning_rate=cfg.lr)  # 创建随机梯度下降优化器，传入可训练参数和学习率

iters_per_epoch = train_dataset.get_dataset_size()  # 获取每个epoch的迭代次数

total_train_steps = iters_per_epoch * cfg.epochs    # 计算总的训练步数

print('iters_per_epoch: ', iters_per_epoch)         # 打印每个epoch的迭代次数

print('total_train_steps: ', total_train_steps)     # 打印总的训练步数

metrics_name = ["acc", "iou", "dice", "sens", "spec"]  # 定义评估指标的名称

best_iou = 0                                           # 初始化最佳iou为0

ckpt_path = 'checkpoint/best_UNet.ckpt'                # 设置保存最佳模型的路径

for epoch in range(cfg.epochs):                 # 遍历每个epoch

    print(f"Epoch [{epoch+1} / {cfg.epochs}]")  # 打印当前epoch的信息

    train(net, train_dataset, criterion, optimizer, metrics_name)           # 在训练集上进行训练

    checkpoint_best, spec = val(net, val_dataset, criterion, metrics_name)  # 在验证集上进行评估，获取最佳检查点和特异性指标

    if epoch > 2 and spec > 0.2:                                                     # 如果当前epoch大于2且特异性指标大于0.2

        if checkpoint_best > best_iou:                                               # 如果最佳检查点的交并比大于当前最佳交并比

            print('IoU improved from %0.4f to %0.4f' % (best_iou, checkpoint_best))  # 打印交并比改善的信息

            best_iou = checkpoint_best                                     # 更新最佳交并比

            mindspore.save_checkpoint(net, ckpt_path)                      # 保存最佳模型的检查点

            print("saving best checkpoint at: {} ".format(ckpt_path))      # 打印保存检查点的路径

        else:

            print('IoU did not improve from %0.4f' % (best_iou),"\n-------------------------------")  # 打印交并比未改善的信息

print("Done!")  # 打印训练完成的信息
```


## 5. 模型预测

```
def val_transforms(img_size):

    return C_transforms.Compose([

        vision_C.Resize(img_size, interpolation=Inter.NEAREST),  # 调整图像大小为指定的img_size，插值方式为最近邻插值

        vision_C.Rescale(1/255., 0),  # 将像素值缩放到范围 [0, 1]，将输入图像除以255

        vision_C.HWC2CHW()  # 将图像的通道维度从 "HWC"（高度、宽度、通道）顺序转换为 "CHW"（通道、高度、宽度）顺序

    ])

class Data_Loader:

    def __init__(self, data_path, have_mask):

        # 初始化函数，读取所有data_path下的图片

        self.data_path = data_path  # 数据集路径

        self.have_mask = have_mask  # 是否有掩码

        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))    # 获取所有图像文件的路径

        if self.have_mask:

            self.label_path = glob.glob(os.path.join(data_path, 'mask/*.png')) # 获取所有标签文件的路径

    def __getitem__(self, index):

        # 根据index读取图片

        image = cv2.imread(self.imgs_path[index])# 读取图像

        if self.have_mask:

            label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)# 读取灰度图像标签

            label = label.reshape((label.shape[0], label.shape[1], 1))      # 将标签的形状调整为 (H, W, 1)

        else:

            label = image                        # 如果没有标签，则将图像作为标签

        return image, label

    @property

    def column_names(self):

        column_names = ['image', 'label']        # 定义列名

        return column_names

    def __len__(self):

        return len(self.imgs_path)               # 返回数据集的长度

def create_dataset(data_dir, img_size, batch_size, shuffle, have_mask=False):

    mc_dataset = Data_Loader(data_path=data_dir, have_mask=have_mask)  # 创建Data_Loader对象，加载数据集

    print(len(mc_dataset))                                             # 打印数据集中的图像文件数量

    # 创建GeneratorDataset对象，使用Data_Loader对象作为数据源

    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)  

    transform_img = val_transforms(img_size)                           # 创建图像数据的转换操作

    seed = random.randint(1, 1000)                                     # 生成随机种子

    mindspore.set_seed(seed)                                           # 设置随机种子

    dataset = dataset.map(input_columns='image', num_parallel_workers=1, operations=transform_img)  # 对图像数据应用转换操作

    mindspore.set_seed(seed)                                           # 设置随机种子

    dataset = dataset.map(input_columns="label", num_parallel_workers=1, operations=transform_img)  # 对标签数据应用转换操作

    dataset = dataset.batch(batch_size, num_parallel_workers=1)        # 批量化数据

    return dataset                                                     # 返回创建的数据集对象

def model_pred(model, test_loader, result_path, have_mask):

    model.set_train(False)  # 设置模型为推理模式

    test_pred = []          # 存储预测结果

    test_label = []         # 存储标签数据

    for batch, (data, label) in enumerate(test_loader.create_tuple_iterator()):

        pred = model(data)  # 使用模型进行预测

        pred[pred > 0.5] = float(1)   # 将预测结果大于0.5的像素置为1

        pred[pred <= 0.5] = float(0)  # 将预测结果小于等于0.5的像素置为0

        preds = np.squeeze(pred, axis=0)      # 去除预测结果的批次维度

        img = np.transpose(preds, (1, 2, 0))  # 转换预测结果的通道维度顺序为"HWC"

        if not os.path.exists(result_path):

            os.makedirs(result_path)          # 创建保存结果的文件夹

        cv2.imwrite(os.path.join(result_path, "%05d.png" % batch), img.asnumpy() * 255.)  # 保存预测结果为图像文件

        test_pred.extend(pred.asnumpy())      # 将预测结果添加到test_pred列表中

        test_label.extend(label.asnumpy())    # 将标签数据添加到test_label列表中

    if have_mask:

        mtr = ['acc', 'iou', 'dice', 'sens', 'spec']  # 定义评估指标

        metric = metrics_(mtr, smooth=1e-5)           # 创建评估指标的计算对象

        metric.clear()                                # 清除评估指标的历史数据

        metric.update(test_pred, test_label)          # 更新评估指标计算结果

        res = metric.eval()                           # 获取评估指标的结果

        # 打印评估指标结果

        print(f'丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))  

    else:

        # 如果没有标签数据，则无法计算评估指标

        print("Evaluation metrics cannot be calculated without Mask")  

if __name__ == '__main__':

    # 创建一个UNet模型对象net，输入通道数为3，输出通道数为1

    net = UNet(3, 1)

    mindspore.load_checkpoint("./Unet/checkpoint/best_UNet.ckpt", net=net)

    #保存预测结果路径为"predict"

    result_path = "predict"

    #创建一个测试数据集加载器test_dataset

    test_dataset = create_dataset("./Unet/ISBI/test/", 224, 1, shuffle=False, have_mask=False)

    #根据net模型进行预测，并将预测结果保存在"predict"目录下

model_pred(net, test_dataset, result_path, have_mask=False)
```

## 6. 预测 mask 可视化

 ```
 image_path = "ISBI/test/image/"

pred_path = "predict/"

image_list = os.listdir(image_path)#读取测试图像

pred_list = os.listdir(pred_path)[1:]#读取预测结果

# print(image_list)

# print(pred_list)

#读取图像文件

test_image = np.array([cv2.imread(image_path + image_list[p], -1) for p in range(len(image_list))])

pred_masks = np.array([cv2.imread(pred_path + pred_list[p], -1) for p in range(len(pred_list))])

#显示测试图像和预测结果。num参数表示要显示的图像数量

show_image(test_image, num = 12)

show_image(pred_masks, num = 12)
 ```