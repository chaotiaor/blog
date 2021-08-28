# fast rcnn

1, 原理

2，改进点

3，loss

# 原理

如下图所示：

![image](https://user-images.githubusercontent.com/37278270/131208869-e62e6892-b287-4e5e-aa3e-12c751c3870e.png)

# 改进点

1, 实现大部分end-to-end训练(提proposal阶段除外)： 所有的特征都暂存在显存中，就不需要额外的磁盘空间

2，提出了一个RoI层，算是SPP的变种，SPP是pooling成多个固定尺度，RoI只pooling到单个固定的尺度

3，[ROI](./FasterRcnn/two.md)

# loss









