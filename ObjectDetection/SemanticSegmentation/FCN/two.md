上采样
===

1， 原理


原理
===
1， Resize，如双线性插值直接缩放，类似于图像缩放
2， Deconvolution，也叫Transposed Convolution


而对于反卷积，相当于把普通卷积反过来，输入蓝色2x2矩阵（周围填0变成6x6），卷积核大小还是3x3。
当设置反卷积参数pad=0，stride=1时输出绿色4x4矩阵， 如下图：

![image](https://user-images.githubusercontent.com/37278270/131640915-a3e40ee1-8827-4625-8137-99324da70a78.png)






参考
===

https://zhuanlan.zhihu.com/p/31428783
























