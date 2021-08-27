stylegan2记录
===

1，网络设计

2, 其他优化

3，评价


网络设计
===
因为StyleGAN存在瑕疵，少量生成的图片有明显的水珠，这个水珠也存在于feature map上，如下图：

![image](https://user-images.githubusercontent.com/37278270/131093640-895a967f-4130-4228-a36f-1a91990af223.png)

导致水珠的原因是 Adain 操作，Adain对每个feature map进行归一化，因此有可能会破坏掉feature之间的信息。是实验证明 当去除Adain的归一化操作后，水珠就消失了。

![image](https://user-images.githubusercontent.com/37278270/131093793-b917c0cc-2d39-4c78-8dc2-8286cb05e8c3.png)

![image](https://user-images.githubusercontent.com/37278270/131093816-dc449689-5397-458b-8d06-9adce2c513c8.png)

上面两张图说明了从 styleGAN 到 styleGAN2 ，在网络结构上的变换，去除normalization之后水珠消失了，但是styleGAN的一个亮点是 style mixing，仅仅只改网络结构，虽然能去除水珠，但是无法对style mixing 有 scale-specific级别的控制

除了网络结构上的改进，还有就是 Weight demodulation，公式如下：

![image](https://user-images.githubusercontent.com/37278270/131094001-f191a885-4b16-404f-aaa6-3acb49b5e728.png)

![image](https://user-images.githubusercontent.com/37278270/131094050-06b65fba-24b5-4d17-a715-c2eda05ffa83.png)

改进后的效果如下，水珠消失了。

![image](https://user-images.githubusercontent.com/37278270/131094119-0d0f0d00-18b4-4df1-866f-d6a3c84277c1.png)



其他优化
===

Lazy regularization

损失是由损失函数和正则项组成，优化的时候也是同时优化这两项的，lazy regularization就是正则项可以减少优化的次数，比如每16个minibatch才优化一次正则项，这样可以减少计算量，同时对效果也没什么影响。

Path length regularization

在生成人脸的同时，我们希望能够控制人脸的属性，不同的latent code能得到不同的人脸，当确定latent code变化的具体方向时，该方向上不同的大小应该对应了图像上某一个具体变化的不同幅度。为了达到这个目的，设计了 Path length regularization ，它的原理也很简单，在图像上的梯度 用 图像乘上变换的梯度 来表示，下列公式中 w 表示由latent code z 得到的disentangled latent code， y 表示图像，这个图像的像素是符合正态分布的， Jw 是生成器 g 对 w 的一阶矩阵，表示图像在 w 上的变化， a 是  ||Jtw y ||2 动态的移动平均值，随着优化动态调整，自动找到一个全局最优值。

![image](https://user-images.githubusercontent.com/37278270/131102556-e4f048c8-8a5a-4056-a9eb-2227760cdc33.png)


No Progressive growth

StyleGAN使用的Progressive growth会有一些缺点，如下图，当人脸向左右偏转的时候，牙齿却没有偏转，即人脸的一些细节如牙齿、眼珠等位置比较固定，没有根据人脸偏转而变化，造成这种现象是因为采用了Progressive growth训练，Progressive growth是先训练低分辨率，等训练稳定后，再加入高一层的分辨率进行训练，训练稳定后再增加分辨率，即每一种分辨率都会去输出结果，这会导致输出频率较高的细节，如下图中的牙齿，而忽视了移动的变化

![image](https://user-images.githubusercontent.com/37278270/131102722-deecd88f-0a9e-4fc5-b505-1aa44dda8204.png)

使用Progressive growth的原因是高分辨率图像生成需要的网络比较大比较深，当网络过深的时候不容易训练，但是skip connection可以解决深度网络的训练，因此有了下图中的三种网络结构，都采用了skip connection，三种网络结构的效果也进行了实验评估，如下下图。

![image](https://user-images.githubusercontent.com/37278270/131103119-7bb0d8b5-51e8-4398-bacc-9caf03307ddf.png)

How to project image to latent code

具体可以查看论文《Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?》



评价
===
我就俩个字， 牛逼


参考文件
===
https://zhuanlan.zhihu.com/p/263554045

https://arxiv.org/abs/1912.04958


