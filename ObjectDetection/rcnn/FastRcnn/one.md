# fast rcnn

1, 原理

2, 改进点

3, loss

# 原理

如下图所示：

![image](https://user-images.githubusercontent.com/37278270/131208869-e62e6892-b287-4e5e-aa3e-12c751c3870e.png)

# 改进点

1, 实现大部分end-to-end训练(提proposal阶段除外)： 所有的特征都暂存在显存中，就不需要额外的磁盘空间

2，提出了一个RoI层，算是SPP的变种，SPP是pooling成多个固定尺度，RoI只pooling到单个固定的尺度

3，[ROI](./two.md)

# loss

如下图：

![image](https://user-images.githubusercontent.com/37278270/131210418-37864468-0ed9-4e30-91e5-85b070e8d5b3.png)

cls_score层用于分类，输出K+1维数组p，表示属于K类和背景的概率。
bbox_prdict层用于调整候选区域位置，输出4 * K维数组t，表示分别属于K类时，应该平移缩放的参数。

其他见

https://blog.csdn.net/shenxiaolu1984/article/details/51036677

代价函数部分



# 参考

https://zhuanlan.zhihu.com/p/24780395

https://blog.csdn.net/shenxiaolu1984/article/details/51036677

Girshick, Ross. “Fast r-cnn.” Proceedings of the IEEE International Conference on Computer Vision. 2015.




