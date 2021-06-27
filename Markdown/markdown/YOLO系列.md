# YOLO系列

## YOLOv1

1、将候选框和分类作为一个任务，使用同一框架进行

2、 YOLOv1采用的是“分而治之”的策略，将一张图片平均分成7×7个网格，每个网格分别负责预测中心点落在该网格内的目标。回忆一下，在Faster R-CNN中，是通过一个RPN来获得目标的感兴趣区域，这种方法精度高，但是需要额外再训练一个RPN网络，这无疑增加了训练的负担。在YOLOv1中，通过划分得到了7×7个网格，这49个网格就相当于是目标的感兴趣区域。通过这种方式，我们就不需要再额外设计一个RPN网络，这正是YOLOv1作为单阶段网络的简单快捷之处！
3、具体实现过程如下：

将一幅图像分成 S×S个网格（grid cell），如果某个 object 的中心落在这个网格中，则这个网格就负责预测这个object。
每个网格要预测 B 个bounding box，每个 bounding box 要预测 (x, y, w, h) 和 confidence 共5个值。
每个网格还要预测一个类别信息，记为 C 个类。
总的来说，S×S 个网格，每个网格要预测 B个bounding box ，还要预测 C 个类。网络输出就是一个 S × S × (5×B+C) 的张量。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722175311393.png#pic_center)

在实际过程中，YOLOv1把一张图片划分为了7×7个网格，并且每个网格预测2个Box（Box1和Box2），20个类别。所以实际上，S=7，B=2，C=20。那么网络输出的shape也就是：7×7×30。

（2）目标损失函数

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722180056692.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dqaW5qaWU=,size_16,color_FFFFFF,t_70#pic_center)

损失由三部分组成，分别是：坐标预测损失、置信度预测损失、类别预测损失。
使用的是差方和误差。需要注意的是，w和h在进行误差计算的时候取的是它们的平方根，原因是对不同大小的bounding box预测中，相比于大bounding box预测偏一点，小box预测偏一点更不能忍受。而差方和误差函数中对同样的偏移loss是一样。 为了缓和这个问题，作者用了一个比较取巧的办法，就是将bounding box的w和h取平方根代替原本的w和h。
定位误差比分类误差更大，所以增加对定位误差的惩罚，使λ coord = 5 λ_{coord}=5λ 
coord =5。
在每个图像中，许多网格单元不包含任何目标。训练时就会把这些网格里的框的“置信度”分数推到零，这往往超过了包含目标的框的梯度。从而可能导致模型不稳定，训练早期发散。因此要减少了不包含目标的框的置信度预测的损失，使 λ n o o b j = 0.5 λ_{noobj}=0.5λ 
noobj=0.5。

## YOLOv2

1、每个卷积层后增加BN（mAP提升了2%），并且去除了Dropout

2、使用高分辨率的数据集在预训练的CNN上做微调（mAP提升了4%）

3、不对（x,y,w,h）进行训练，而是预测与Anchor框的偏差（offset），每个格点指定n个Anchor框，最接近GT的框产生loss，其余不产生loss

4、区别于FasterRCNN，Anchor Box的宽高不是认为设定，而是将训练数据集中的矩形框的宽高，用Kmeans聚类得到先验框的宽和高。例如使用5个AnchorBox，那么Kmeans聚类的中心个数设置为5。

5、为了不损失细粒度特征，在passthrough层将26*26*1的特征图，变成13\*13\*4的特征图

6、由于没有FC层，YOLOv2可以接受Multi-Scale Training（多尺度训练）

​		原图resize作卷积，这样的一个金字塔（不如FPN）

​		每10个epoch，将图片resize成（320,352，...，608）中的一种

​		如果Anchor Box设置为5

​		320*320输出格点是10\*10共预测500个 结果

​		608*608输出格点是19\*19共预测1805个结果

## YOLOv3

改进小物体检测

![0_20200411132534_478.jpeg](https://imgconvert.csdnimg.cn/aHR0cDovL3d3dy5jaGVuamlhbnF1LmNvbS9tZWRpYS91cGltZy8wXzIwMjAwNDExMTMyNTM0XzQ3OC5qcGVn?x-oss-process=image/format,png)

FPN

**主要模块：**

​	**Darknet**：开源框架

​	**DBL**：DarknetConv2D+BN+LeakyReLu

​	**Resunit**: DBL+DBL+残差

​	**Resblock**: zeropadding + DBL + Resunit*n

255 = 3\*(4+1+80)

输出通道255  3表示一个格子的有三个bounding box, 4 表示框的4个坐标信息，1表示有无物体，80表示类别

## YOLOv4

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20200831113450308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25hbjM1NTY1NTYwMA==,size_16,color_FFFFFF,t_70#pic_center)

**Yolov4**的结构图和**Yolov3**相比，因为多了**CSP结构**，**PAN结构**，如果单纯看可视化流程图，会觉得很绕，但是在绘制出上面的图形后，会觉得豁然开朗，其实整体架构和Yolov3是相同的，不过使用各种新的算法思想对各个子结构都进行了改进。

先整理下Yolov4的五个基本组件：
1. **CBM**：Yolov4网络结构中的最小组件，由Conv+Bn+Mish激活函数三者组成。
2. **CBL**：由Conv+Bn+Leaky_relu激活函数三者组成。
3. **Res unit**：借鉴Resnet网络中的残差结构，让网络可以构建的更深。
4. **CSPX**：借鉴CSPNet网络结构，由卷积层和X个Res unint模块Concat组成。
5. **SPP**：采用1×1，5×5，9×9，13×13的最大池化的方式，进行多尺度融合。

其他基础操作：

1. Concat：张量拼接，维度会扩充，和Yolov3中的解释一样，对应于cfg文件中的route操作。
2. Add：张量相加，不会扩充维度，对应于cfg文件中的shortcut操作。

Backbone中卷积层的数量：
和Yolov3一样，再来数一下Backbone里面的卷积层数量。
每个CSPX中包含5+2×X个卷积层，因此整个主干网络Backbone中一共包含1+（5+2×1）+（5+2×2）+（5+2×8）+（5+2×8）+（5+2×4）=72。

## 4.3 核心基础内容
Yolov4本质上和Yolov3相差不大，可能有些人会觉得失望。
但我觉得算法创新分为三种方式：
（1）第一种：面目一新的创新，比如Yolov1、Faster-RCNN、Centernet等，开创出新的算法领域，不过这种也是最难的。
（2）第二种：守正出奇的创新，比如将图像金字塔改进为特征金字塔。
（3）第三种：各种先进算法集成的创新，比如不同领域发表的最新论文的tricks，集成到自己的算法中，却发现有出乎意料的改进。
Yolov4既有第二种也有第三种创新，组合尝试了大量深度学习领域最新论文的20多项研究成果，而且不得不佩服的是作者Alexey在Github代码库维护的频繁程度。
目前Yolov4代码的Star数量已经1万左右，据我所了解，目前超过这个数量的，目标检测领域只有Facebook的Detectron(v1-v2)、和Yolo(v1-v3)官方代码库（已停止更新）。
所以Yolov4中的各种创新方式，大白觉得还是很值得仔细研究的。

为了便于分析，将Yolov4的整体结构拆分成四大板块：
大白主要从以上4个部分对YoloV4的创新之处进行讲解，让大家一目了然。

（1）输入端：这里指的创新主要是训练时对输入端的改进，主要包括Mosaic数据增强、cmBN、SAT自对抗训练。
（2）BackBone主干网络：将各种新的方式结合起来，包括：CSPDarknet53、Mish激活函数、Dropblock
（3）Neck：目标检测网络在BackBone和最后的输出层之间往往会插入一些层，比如Yolov4中的SPP模块、FPN+PAN结构
（4）Prediction：输出层的锚框机制和Yolov3相同，主要改进的是训练时的损失函数CIOU_Loss，以及预测框筛选的nms变为DIOU_nms

## YOLOv5