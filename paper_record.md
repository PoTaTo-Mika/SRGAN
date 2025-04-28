### 记录

现在我要做的是以下三个实验了：

1.把FCA嵌入到ResNet里面，还是ResidualBlock但是存在一个FCA在ResidualBlock类里面。

2.把ResNet嵌入到FCA里面，然后我们用FCALayer去替换原有的所有ResidualBlock。

3.把PCA也加入到FCA里面，然后按照1的实验再做一次，因为1的效果之前看下来是还可以的。

4.要不要按照3把2做一次，待定。

然后我们细分一下任务：

1.用现有的权重做一点超分的实验，用digital typhoon（主要任务）

2.跑其它的GAN的结果，打算用现有权重先试着，需要炼就再说。