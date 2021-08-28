# rcnn

0, 常用概念

1, 原理

2, 意义

3, 训练方法

# 常用概念

1, Bounding Box(bbox)

bbox是包含物体的最小矩形，该物体应在最小矩形内部, 
物体检测中关于物体位置的信息输出是一组(x,y,w,h)数据，
其中x,y代表着bbox的左上角(或者其他固定点，可自定义)，对应的w,h表示bbox的宽和高.一组(x,y,w,h)可以唯一的确定一个定位框。

2, Intersection over Union(IoU)

对于两个区域R和R′,则两个区域的重叠程度overlap计算如下:

O(R,R′)=|R∩R′|/|R∪R′|

如下图：

![image](https://user-images.githubusercontent.com/37278270/131203527-da74d627-d0ef-4e20-9b0d-1bcb5d68e218.png)

3， 非极大值抑制(Non-Maximum Suppression又称NMS)

非极大值抑制，简称为NMS算法，英文为Non-Maximum Suppression。其思想是搜素局部最大值，抑制极大值。NMS算法在不同应用中的具体实现不太一样，但思想是一样的。

使用方法：

前提：目标边界框列表及其对应的置信度得分列表，设定阈值，阈值用来删除重叠较大的边界框。

IoU：intersection-over-union，即两个边界框的交集部分除以它们的并集。

非极大值抑制的流程如下：

根据置信度得分进行排序

选择置信度最高的比边界框添加到最终输出列表中，将其从边界框列表中删除

计算所有边界框的面积

计算置信度最高的边界框与其它候选框的IoU。

删除IoU大于阈值的边界框

重复上述过程，直至边界框列表为空


# 原理

如下图：

![image](https://user-images.githubusercontent.com/37278270/131203707-7cd62a03-4cc2-48ec-bd6d-9981f2653f3c.png)


借鉴了滑动窗口思想，R-CNN 采用对区域进行识别的方案。

具体是：

1, 给定一张输入图片，从图片中提取 2000 个类别独立的候选区域。

2, 对于每个区域利用 CNN 抽取一个固定长度的特征向量。

3, 再对每个区域利用 SVM 进行目标分类。

如下图：

![image](https://user-images.githubusercontent.com/37278270/131203795-b9c6a1e0-1035-484a-a6fa-d263f779e4d4.png)

# 意义
1, 在 Pascal VOC 2012 的数据集上，能够将目标检测的验证指标 mAP 提升到 53.3%,这相对于之前最好的结果提升了整整 30%.

2, 这一方法证明了可以将神经网络应用在自底向上的候选区域，这样就可以进行目标分类和目标定位。

3, 这一方法也带来了一个观点，那就是当你缺乏大量的标注数据时，比较好的可行的手段是，进行神经网络的迁移学习，采用在其他大型数据集训练过后的神经网络，
然后在小规模特定的数据集中进行 fine-tune 微调。


# 训练方法
具体查看

https://www.jianshu.com/p/5056e6143ed5

训练方法部分


# 参考
https://blog.csdn.net/briblue/article/details/82012575

https://arxiv.org/abs/1311.2524

https://www.jianshu.com/p/5056e6143ed5

https://www.jianshu.com/p/d452b5615850


