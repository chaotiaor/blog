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
===

网络结构包含两个部分，第一个是Mapping network，即下图 (b)中的左部分，由隐藏变量 z 生成 中间隐藏变量 w的过程，这个 w 就是用来控制生成图像的style，即风格。

第二个是Synthesis network，它的作用是生成图像，创新之处在于给每一层子网络都喂了 A 和 B，A 是由 w 转换得到的仿射变换，
用于控制生成图像的风格，B 是转换后的随机噪声，用于丰富生成图像的细节，即每个卷积层都能根据输入的A来调整"style"。整个网络结构还是保持了 PG-GAN （progressive growing GAN） 的结构。

此外，传统的GAN网络输入是一个随机变量或者隐藏变量 z，但是StyleGAN 将 z 单独用 mapping网络将z变换成w，
再将w投喂给 Synthesis network的每一层，因此Synthesis network中最开始的输入变成了常数张量

![image](https://user-images.githubusercontent.com/37278270/130754103-6df91d5b-6271-4eb0-a4c5-c652a2a12069.png)

以下是对每个部分的详细解说


a, Mapping network
===

Mapping network 要做的事就是对隐藏空间（latent space）进行解耦。

latent code是为了更好的对数据进行分类或生成，需要对数据的特征进行表示，但是数据有很多特征，这些特征之间相互关联，耦合性较高，导致模型很难弄清楚它们之间的关联，使得学习效率低下，因此需要寻找到这些表面特征之下隐藏的深层次的关系，将这些关系进行解耦，得到的隐藏特征，即latent code。由 latent code组成的空间就是 latent space。

Mapping network由8个全连接层组成，通过一系列仿射变换，由 z 得到 w，这个 w 转换成风格 y = (Ya, Yb) ，结合 AdaIN (adaptive instance normalization) 风格变换方法：

![image](https://user-images.githubusercontent.com/37278270/130757373-e09cba36-cccb-4edb-a298-f86895a73173.png)

Xi 表示每个特征图。

前面提到 Mapping network 是将 latent code z 变成 w，为什么要把 z 变成 w 呢，一般 z 是符合均匀分布或者高斯分布的随机向量，

但在实际情况中，并不是这样，比如特征：头发的长度 和 男子气概，下图（a）中就是这两个特征的组合，左上角缺失的部分代表头发越长，男子气概越强，如果直接用 均匀分布或者高斯分布对特征变量头发长度和男子气概进行采样，得到的结果都不准确，

因此在（b）中将分布（a）warp 成连续的分布函数 f(z)，这个 f(z) 的密度是非均匀的，图 (c) 是 w 的分布。

![image](https://user-images.githubusercontent.com/37278270/130757833-a4fa0fa0-4730-411a-b748-70e3665760b8.png)


b, latent space interpolations
===

latent space interpolations 不是StyleGAN提到的，但在多篇paper中有提到，如下图的椅子，左边是比较宽的椅子，右边是比较窄的椅子，中间的椅子是这两种椅子特征的线性组合。

![image](https://user-images.githubusercontent.com/37278270/130757976-4878d8dd-ce19-4a3d-9e42-59a8779b4514.png)

人脸的latent space interpolations效果图

![image](https://user-images.githubusercontent.com/37278270/130758035-72fd7b0c-7d5a-4a8a-ae03-d76c0dda560a.png)


c, Style mixing
===

下图中第一行是 source B， 第一列是source A，source A 和 source B的每张图片由各自相应的latent code 生成，剩余的图片是对 source A 和 souce B 风格的组合。 Style mixing 的本意是去找到控制不同style的latent code的区域位置，具体做法是将两个不同的latent code z1 和 z2 输入到 mappint network 中，分别得到 w1 和 w2 ，分别代表两种不同的 style，然后在 synthesis network 中随机选一个中间的交叉点，交叉点之前的部分使用 w1 ，交叉点之后的部分使用 w2 ，生成的图像应该同时具有 source A 和 source B 的特征，称为 style mixing。

根据交叉点选取位置的不同，style组合的结果也不同。下图中分为三个部分，第一部分是 Coarse styles from source B，分辨率(4x4 - 8x8)的网络部分使用B的style，其余使用A的style, 可以看到图像的身份特征随souce B，但是肤色等细节随source A；第二部分是 Middle styles from source B，分辨率(16x16 - 32x32)的网络部分使用B的style，这个时候生成图像不再具有B的身份特性，发型、姿态等都发生改变，但是肤色依然随A；第三部分 Fine from B，分辨率(64x64 - 1024x1024)的网络部分使用B的style，此时身份特征随A，肤色随B。由此可以大致推断，低分辨率的style 控制姿态、脸型、配件 比如眼镜、发型等style，高分辨率的style控制肤色、头发颜色、背景色等style。

![image](https://user-images.githubusercontent.com/37278270/130759040-a1b70689-6f30-4591-bd14-1e31d895dbeb.png)


d, Stochastic variation
===
论文中的 Stochastic variation 是为了让生成的人脸的细节部分更随机、更自然，细节部分主要指头发丝、皱纹、皮肤毛孔、胡子茬等。如下图。

![image](https://user-images.githubusercontent.com/37278270/130760316-fa39a89e-983c-46b8-a154-261ce71e34dd.png)

实现这种 Stochastic variation 的方法就是引入噪声，StyleGAN的做法是在每一次卷积操作后都加入噪声，下图是不同网络层加入噪声的对比。

![image](https://user-images.githubusercontent.com/37278270/130760532-96c6d1f9-9330-4d7a-b614-cff439c9cea2.png)


e, Perceptual path length
===
图像生成其实是学习从一个分布到目标分布的迁移过程，如下图，已知input latent code 是z1，或者说白色的狗所表示的latent code是z1，目标图像是黑色的狗，黑狗图像的latent code 是 z2，图中蓝色的虚线是z1 到 z2 最快的路径，绿色的曲线是我们不希望的路径，在蓝色的路径中的中间图像应该是z1 和 z2 的组合，假设这种组合是线性的（当特征充分解耦的时候），蓝色路径上生成的中间图像也是狗（ 符合 latent-space interpolation），但是绿色的曲线由于偏离路径太多，生成的中间图像可能是其他的，比如图上的卧室，这是我们不希望的结果。

补充一下，我们可以通过训练好的生成模型得到给定图像的latent code，假设我们有一个在某个数据集上训练好的styleGAN模型，现在要找到一张图像 x 在这个模型中的latent code，设初始latent code 是 z，生成的初始图像是p，通过 p 和 x 之间的差距 设置损失函数，通过损失不断去迭代 z，最后得到图像x的latent code。

![image](https://user-images.githubusercontent.com/37278270/130760734-f9721efd-c925-4e67-ab0c-a7c9b362177d.png)

Perceptual path length 是一个指标，用于判断生成器是否选择了最近的路线（比如上图蓝色虚线），用训练过程中相邻时间节点上的两个生成图像的距离来表示，公式如下：

![image](https://user-images.githubusercontent.com/37278270/130761404-240065cb-2019-4e27-b643-edd0501b45eb.png)

g 表示生成器，d表示d(·, ·) evaluates the perceptual distance between the resulting images， f 表示mapping netwrok， 

f(z1)表示由latent code z1 得到的中间隐藏码 w ， t 表示某一个时间点， t属于[0, 1] , t+小量 表示下一个时间点，lerp 表示线性插值 （linear interpolation），即在 latent space上进行插值。

![image](https://user-images.githubusercontent.com/37278270/130761403-7fc3cd5a-1c81-413e-88d8-60d67e56d2b8.png)


g, Truncation Trick
===

Truncation Trick 不是StyleGAN提出来的，它很早就在GAN里用于图像生成了，感兴趣的可以追踪溯源。从数据分布来说，低概率密度的数据在网络中的表达能力很弱，直观理解就是，低概率密度的数据出现次数少，能影响网络梯度的机会也少，但并不代表低概率密度的数据不重要。可以提高数据分布的整体密度，把分布稀疏的数据点都聚拢到一起，类似于PCA，做法很简单，首先找到数据中的一个平均点，然后计算其他所有点到这个平均点的距离，对每个距离按照统一标准进行压缩，这样就能将数据点都聚拢了，但是又不会改变点与点之间的距离关系。

公式如下：

![image](https://user-images.githubusercontent.com/37278270/130762175-def2c2c0-45e0-4200-83c4-163c63cc7489.png)

![](http://latex.codecogs.com/gif.latex?\\psi)是一个实数，表示压缩倍数，下图是truncation对style的影响。

![image](https://user-images.githubusercontent.com/37278270/130763390-646229c7-3b19-458a-b24c-8e13aab82148.png)

2,判别器
===
可以自行设计


loss设计
===
可以自行设计


评价
===
1，在生成大图质量上是十分出色的

2，对设备要求较高


参考文件：
===
https://zhuanlan.zhihu.com/p/263554045

https://arxiv.org/pdf/1812.04948.pdf






