MaskRcnn
===

1, 原理

2，loss


原理
===

如下图:

![image](https://user-images.githubusercontent.com/37278270/131234849-ef6da8f2-eed1-4cf0-b31d-e3ea5f5fe565.png)

其中黑色部分为原来的Faster-RCNN，红色部分为在Faster-RCNN网络上的修改。

将RoI Pooling 层替换成了RoIAlign层；添加了并列的FCN层（mask层）。

具体结构如下图:

![image](https://user-images.githubusercontent.com/37278270/131235615-fe7f1f6c-c5d3-4a77-b929-8d96f104281f.png)

包括以下几个部分:

1,  [FPN](https://github.com/chaotiaor/blog/blob/master/ObjectDetection/rcnn/FPN/one.md)

2,  [RPN](https://github.com/chaotiaor/blog/blob/master/ObjectDetection/rcnn/FasterRcnn/two.md)

3,  [RoIAlign](./two.md)

总结如下：

1， 骨干网络ResNet-FPN，用于特征提取，另外，ResNet还可以是：ResNet-50,ResNet-101,ResNeXt-50,ResNeXt-101；

1， 头部网络，包括边界框识别（分类和回归）+mask预测。头部结构见下图：

![image](https://user-images.githubusercontent.com/37278270/131235744-a90342c5-5480-4cd4-8f3c-10d05a18baaa.png)



loss
===

![image](https://user-images.githubusercontent.com/37278270/131235856-a8ffb0e2-1982-4d6a-94be-450aeb8b0f25.png)

具体参考
https://zhuanlan.zhihu.com/p/57759536
损失函数部分



参考
===

https://arxiv.org/abs/1703.06870

https://zhuanlan.zhihu.com/p/57759536

https://zhuanlan.zhihu.com/p/37998710

https://blog.csdn.net/qq_37392244/article/details/88844681
















