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

1, [RoIAlign](./two.md)

2, [FPN]()





参考
===

https://arxiv.org/abs/1703.06870

https://zhuanlan.zhihu.com/p/57759536

https://zhuanlan.zhihu.com/p/37998710

https://blog.csdn.net/qq_37392244/article/details/88844681
















