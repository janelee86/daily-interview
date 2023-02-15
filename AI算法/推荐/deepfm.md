# DeepFM





DeepFM模型是2017年由哈工大与华为联合提出的模型，是对Wide&Deep模型的改进。与DCN不同的是，DeepFM模型是将Wide部分替换为了FM模型，增强了模型的低阶特征交互的能力。关于低阶特征交互，文章的Introduction中也提到了其重要性，例如：

1、用户经常在饭点下载送餐APP，故存在一个2阶交互：app种类与时间戳；

2、青少年喜欢射击游戏和RPG游戏，存在一个3阶交互：app种类、用户性别和年龄；

用户背后的特征交互非常的复杂，低阶和高阶的特征交互都是很重要的，这也证明了Wide&Deep这种模型架构的有效性。DeepFM是一种**端到端的模型**，强调了包括低阶和高阶的特征交互接下来直接对DeepFM模型架构进行介绍，并与其他之前提到过的模型进行简单的对比。


The DeepFM model is a model jointly proposed by Harbin Institute of Technology and Huawei in 2017, which is an improvement of the Wide&Deep model. Different from DCN, the DeepFM model replaces the Wide part with the FM model, which enhances the low-order feature interaction capability of the model. Regarding low-level feature interaction, its importance is also mentioned in the Introduction of the article, for example:

1. Users often download food delivery apps at meal times, so there is a second-order interaction: app type and timestamp;

2. Teenagers like shooting games and RPG games, and there is a third-order interaction: app type, user gender and age;

The feature interaction behind the user is very complex, and both low-order and high-order feature interactions are very important, which also proves the effectiveness of the Wide&Deep model architecture. DeepFM is an **end-to-end model**, which emphasizes the interaction of low-level and high-level features. Next, the DeepFM model architecture will be introduced directly, and a simple comparison with other previously mentioned models will be made.


## 模型结构

DeepFM的模型结构非常简单，由Wide部分与Deep部分共同组成，如下图所示：
The model structure of DeepFM is very simple, consisting of a Wide part and a Deep part, as shown in the figure below:
<img src="http://gzy-gallery.oss-cn-shanghai.aliyuncs.com/work_img/21.png" style="zoom: 50%;" />

在论文中模型的目标是**共同学习低阶和高阶特征交互**，应用场景依旧是CTR预估，因此是一个二分类任务（$y=1$表示用户点击物品，$y=0$则表示用户未点击物品）

The goal of the model in the paper is to **jointly learn low-level and high-level feature interactions**, the application scenario is still CTR prediction, so it is a binary classification task ($y=1$ means the user clicks on the item, $y=0 $ indicates that the user did not click on the item)
### Input与Embedding层

关于输入，包括离散的分类特征域（如性别、地区等）和连续的数值特征域（如年龄等）。分类特征域一般通过one-hot或者multi-hot（如用户的浏览历史）进行处理后作为输入特征；数值特征域可以直接作为输入特征，也可以进行离散化进行one-hot编码后作为输入特征。
对于每一个特征域，需要单独的进行Embedding操作，因为每个特征域几乎没有任何的关联，如性别和地区。而数值特征无需进行Embedding。
Regarding the input, it includes discrete categorical feature fields (such as gender, region, etc.) and continuous numerical feature fields (such as age, etc.). Categorical feature domains are generally processed as input features through one-hot or multi-hot (such as user browsing history); numerical feature domains can be directly used as input features, or discretized and one-hot encoded as input features.
For each feature domain, an Embedding operation needs to be performed separately, because each feature domain has almost no correlation, such as gender and region. The numerical features do not need to be Embedding.
Embedding结构如下：

<img src="http://gzy-gallery.oss-cn-shanghai.aliyuncs.com/work_img/22.png" style="zoom: 50%;" />



文章中指出每个特征域使用的Embedding维度$k$都是相同的。

【注】与Wide&Deep不同的是，DeepFM中的**Wide部分与Deep部分共享了输入特征**，即Embedding向量。
[Note] Unlike Wide&Deep, the **Wide part and Deep part in DeepFM share the input feature**, that is, the Embedding vector.


#### Wide部分---FM

<img src="http://gzy-gallery.oss-cn-shanghai.aliyuncs.com/work_img/23.png" style="zoom:67%;" />



FM模型[^4]是2010年Rendle提出的一个强大的**非线性分类模型**，除了特征间的线性(1阶)相互作用外，FM还将特征间的(2阶)相互作用作为各自特征潜向量的内积进行j建模。通过隐向量的引入使得FM模型更好的去处理数据稀疏行的问题，想具体了解的可以看一下原文。DeepFM模型的Wide部分就直接使用了FM，Embedding向量作为FM的输入。
The FM model is a powerful **nonlinear classification model** proposed by Rendle in 2010. In addition to the linear (1st order) interaction between features, FM also takes the (2nd order) interaction between features as The inner product of the respective feature latent vectors is modeled by j. Through the introduction of hidden vectors, the FM model can better deal with the problem of sparse rows of data. If you want to know more about it, you can read the original text. The Wide part of the DeepFM model directly uses FM, and the Embedding vector is used as the input of FM.

具体的对于2阶特征，FM论文中有下述计算（采取原文的描述形式），为线性复杂复杂度$O(kn)$：
Specifically for the second-order features, the FM paper has the following calculations (in the form of the original description), which is the linear complexity $O(kn)$

#### Deep部分

<img src="http://gzy-gallery.oss-cn-shanghai.aliyuncs.com/work_img/24.png" style="zoom:67%;" />

Deep部分是一个前向传播的神经网络，用来学习高阶特征交互。
The Deep part is a forward propagation neural network used to learn high-level feature interactions.


### Output层

FM层与Deep层的输出相拼接，最后通过一个逻辑回归返回最终的预测结果：
The FM layer is spliced ​​with the output of the Deep layer, and finally returns the final prediction result through a logistic regression:
$$
\hat y=sigmoid(y_{FM}+y_{DNN})
$$



## 面试相关

1、Wide&Deep与DeepFM的区别？
1. What is the difference between Wide&Deep and DeepFM?
Wide&Deep模型，Wide部分采用人工特征+LR的形式，而DeepFM的Wide部分采用FM模型，包含了1阶特征与二阶特征的交叉，且是端到端的，无需人工的特征工程。

In the Wide&Deep model, the Wide part adopts the form of artificial features + LR, while the Wide part of DeepFM adopts the FM model, which includes the intersection of first-order features and second-order features, and is end-to-end without manual feature engineering.


2、DeepFM的Wide部分与Deep部分分别是什么？Embedding内容是否共享
What are the Wide and Deep parts of DeepFM? Whether the Embedding content is shared

Wide：FM，Deep：DNN（deep neural network）；

Embedding内容是共享的，在FM的应用是二阶特征交叉时的表征。
Embedding content is shared, and the application of FM is the representation of the second-order feature intersection.
