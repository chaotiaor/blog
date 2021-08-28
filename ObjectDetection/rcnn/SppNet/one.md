# Spp net

1, 原理


# 原理

为了使得CNN可以接受多尺度输入，我们把SPP(Spatial Pyramid Pooling)层加到FC layer之前即可。

如下图所示，

第一行是原来的CNN网络，需要crop/warp输入图片；

第二行是加了SPP层的CNN网络，可以接受任何size的输入。

![image](https://user-images.githubusercontent.com/37278270/131209237-2779d965-313e-4672-a35d-ba677e376007.png)

所以，如果我们在R-CNN的conv5层之后加入SPP layer，那对于不同size的region proposal的feature map就不需要再进行warp了，直接可以进行分类了。

如图所示，不同size的feature map经过SPP后都变成固定长度的。

![image](https://user-images.githubusercontent.com/37278270/131209288-612c35a2-3d2c-4a45-801c-eb42509d8d81.png)

具体来说就是把输入的feature map划分成不同尺度的，比如图中(4, 4) (2, 2) (1, 1)三种不同的尺度，然后会产生不同的bin，

比如分成(4, 4)就16个bin，然后在每个bin中使用max pooling，然后就变成固定长度为16的向量。

例如下图9和图10中不同尺寸的输入，经过SPP层之后都得到了相同的长度的向量，之后再输入FC layer就可以啦。

![image](https://user-images.githubusercontent.com/37278270/131209523-44860a90-5dcd-4145-8869-1b5abd44a718.png)

![image](https://user-images.githubusercontent.com/37278270/131209529-53341280-1627-4ba2-9f3b-5f25f57c4ba2.png)


# 参考

https://arxiv.org/abs/1406.4729

https://zhuanlan.zhihu.com/p/60919662






















