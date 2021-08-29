# Region Proposal Networks(RPN)

1, 原理


# 原理

如下图：

![image](https://user-images.githubusercontent.com/37278270/131211175-983ee346-e251-43a6-8375-cd0b7272d960.png)

可以看到RPN网络实际分为2条线，
上面一条通过softmax分类anchors获得positive和negative分类，

下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。

而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。

总结如下：

生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals

具体流程：

1， 生成anchors

![image](https://user-images.githubusercontent.com/37278270/131211118-c63ec2b9-7cfc-4a60-a8f0-3905881a87d6.png)

anchors是在每一个特征图上的格点中随机生成的矩形框，论文中生成3种比例的矩形框，加上正负样本，一共18个

2， softmax分类器提取positvie anchors

一副MxN大小的矩阵送入Faster RCNN网络后，到RPN网络变为(M/16)x(N/16)，不妨设 W=M/16，H=N/16。在进入reshape与softmax之前，先做了1x1卷积，如图9：

![image](https://user-images.githubusercontent.com/37278270/131211591-d2f33648-2ad3-4090-9249-ec88d6131cd3.png)



3， bbox reg回归positive anchors

参考https://www.cnblogs.com/bile/p/9117253.html

中3.6.4和3.6.5部分



4， Proposal Layer生成proposals

Proposal Layer有3个输入：

1， positive vs negative anchors分类器结果rpn_cls_prob_reshape

2， im_info

3， 另外还有参数feature_stride=16。

im_info，对于一副任意大小PxQ图像，传入Faster RCNN前首先reshape到固定MxN，im_info=[M, N, scale_factor]则保存了此次缩放的所有信息。
然后经过Conv Layers，经过4次pooling变为WxH=(M/16)x(N/16)大小，其中feature_stride=16则保存了该信息，用于计算anchor偏移量。
如下图：

![image](https://user-images.githubusercontent.com/37278270/131211928-da365b6e-12e8-41fc-a16a-2021efef0b5f.png)

Proposal Layer forward 按照以下顺序依次处理：

1， 生成anchors，利用对预测坐标的导数对所有的anchors做bbox regression回归（这里的anchors生成和训练时完全一致）

2， 按照输入的positive softmax scores由大到小排序anchors，提取前pre_nms_topN(e.g. 6000)个anchors，即提取修正位置后的positive anchors

3， 限定超出图像边界的positive anchors为图像边界，防止后续roi pooling时proposal超出图像边界

4， 剔除尺寸非常小的positive anchors

5， 对剩余的positive anchors进行NMS（nonmaximum suppression）

6， Proposal Layer有3个输入：positive和negative anchors分类器结果rpn_cls_prob_reshape，对应的bbox reg的(e.g. 300)结果作为proposal输出

之后输出proposal=[x1, y1, x2, y2]，注意，由于在第三步中将anchors映射回原图判断是否超出边界，所以这里输出的proposal是对应MxN输入图像尺度的，这点在后续网络中有用


# 参考

https://zhuanlan.zhihu.com/p/31426458

https://www.cnblogs.com/bile/p/9117253.html
























