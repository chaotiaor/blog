FCN
===

1， 原理


原理
===

对于一般的分类CNN网络，如VGG和Resnet，都会在网络的最后加入一些全连接层，经过softmax后就可以获得类别概率信息。
但是这个概率信息是1维的，即只能标识整个图片的类别，不能标识每个像素点的类别，所以这种全连接方法不适用于图像分割。

而FCN提出可以把后面几个全连接都换成卷积，这样就可以获得一张2维的feature map，后接softmax获得每个像素点的分类信息，从而解决了分割问题，如下图

![image](https://user-images.githubusercontent.com/37278270/131634215-de9ee235-5672-4607-bbb2-fcac9c6f943d.png)

结构原理图如下：

![image](https://user-images.githubusercontent.com/37278270/131634491-282f50ea-59cb-4d37-9577-164588147257.png)

3种网络结果对比，明显可以看出效果：FCN-32s < FCN-16s < FCN-8s，即使用多层feature融合有利于提高分割准确性。

![image](https://user-images.githubusercontent.com/37278270/131634769-fbdab101-9d36-4d07-be8f-8f77d7c1bc1d.png)





