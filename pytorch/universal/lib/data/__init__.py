"""
数据模块处理思路如下
1，首先要对数据做一个数据均衡，应该基于之前对数据集的划分，
把每个类的数据打一个标签
2，读取数据之后，对所有的数据进行一个增广，包括2中增广方法，
一是几何角度的，二是视觉角度的，几何角度的需要图像和label一起都是要进行增广
视觉角度的只需对原图进行增广。
3，数据增广后，需要选定合理的归一化参数
"""









