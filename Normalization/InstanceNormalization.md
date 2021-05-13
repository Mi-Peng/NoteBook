## Instance Normalization

### 1. What is Instance Normalization

&emsp;&emsp;仿照BN操作对单独样本进行归一化的IN。




### 2.Difference between IN and BN
&emsp;&emsp;认知任务（分类/分割）用BN，生成任务（style transfer, deblur, GAN）用IN。IN带来了不同sample之间在feature均值方差统计量上的不变性，而BN作为一种global的归一化方式则保留了sample之间的差异性。


### Ref
---
 * [Batch normalization和Instance normalization的?比？](https://www.zhihu.com/question/68730628/answer/937869314)