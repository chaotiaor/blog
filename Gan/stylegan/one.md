stylegan记录
===

1，原理

2，网络设计

3，loss设计

4，评价


原理
===
这个gan是一种无监督的弱生成对抗模型，主要的改动在生成器上，可以通过添加噪音生成不同风格的图像。

StyleGAN中的“Style”是指数据集中人脸的主要属性，比如人物的姿态等信息，而不是风格转换中的图像风格，这里Style是指人脸的风格，包括了脸型上面的表情、人脸朝向、发型等等，
还包括纹理细节上的人脸肤色、人脸光照等方方面面。

StyleGAN 用风格（style）来影响人脸的姿态、身份特征等，用噪声 ( noise ) 来影响头发丝、皱纹、肤色等细节部分。

我在人脸变化的项目中有较多的研究。

网络设计
===
1，生成器

网络结构包含两个部分，第一个是Mapping network，即下图 (b)中的左部分，由隐藏变量 z 生成 中间隐藏变量 w的过程，这个 w 就是用来控制生成图像的style，即风格。

第二个是Synthesis network，它的作用是生成图像，创新之处在于给每一层子网络都喂了 A 和 B，A 是由 w 转换得到的仿射变换，
用于控制生成图像的风格，B 是转换后的随机噪声，用于丰富生成图像的细节，即每个卷积层都能根据输入的A来调整"style"。整个网络结构还是保持了 PG-GAN （progressive growing GAN） 的结构。

此外，传统的GAN网络输入是一个随机变量或者隐藏变量 z，但是StyleGAN 将 z 单独用 mapping网络将z变换成w，
再将w投喂给 Synthesis network的每一层，因此Synthesis network中最开始的输入变成了常数张量

![image](https://user-images.githubusercontent.com/37278270/130754103-6df91d5b-6271-4eb0-a4c5-c652a2a12069.png)





















