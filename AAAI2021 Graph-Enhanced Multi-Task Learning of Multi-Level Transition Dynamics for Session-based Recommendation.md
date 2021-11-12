

Graph-Enhanced Multi-Task Learning of Multi-Level Transition Dynamics for Session-based Recommendation

### 动机：

大多数现有的基于会话的推荐技术并没有很好地设计来捕捉复杂的转换动态(complex transition dynamics)，这些动态表现为时间有序和多层次相互依赖的关系结构。

complex transition dynamics 的"complex"体现在：multi-level relation(intra- and inter-session item relation) . 会话内：short-term and long-term item transition，会话间：long-range cross-session dependencies。复杂依赖的例子见Figure 1.

![image-20210924144140485](.\images\image-20210924144140485.png)

### 开源代码:

https://github.com/sessionRec/MTD

### 主要贡献：

1.开发了一种新的会话推荐框架，可以捕获会话内和会话间的物品转换模式（多层次转换动态）

2.开发了一种位置感知的注意力机，用于学习会话内的序列行为和session-specific knowledge。此外，在图神经网络范例的基础上，建立了全局上下文增强的会话间关系编码器，赋予MTD来捕获会话间项目依赖关系。

3.在三个数据集上取得了SOTA，Yoochoose、Diginetica、Retailrocket。

### 网络图：

![image-20210924155218469](.\images\image-20210924155218469.png)

### 方法论：

#### 1.学习会话内的物品关系

1）Self-attention层：

QKV self-attention + feed forward network

2）位置感知的物品级的注意力聚合模块：
$$
\alpha_i = \delta(\mathbf{g}^T \cdot \sigma(\mathbf{W}_3\cdot\mathbf{x}_{s,I} + \mathbf{W}_4\cdot\mathbf{x}_{s,i}))
\\
\mathbf{x}^*_s = \sum_{i=1}^I\alpha_i\cdot\mathbf{x_\mathit{s,i}}
\\
\mathbf{p}_s = \sum_{i=1}^I\omega_i\cdot\mathbf{x_\mathit{s,i}} \:;\omega_i正比于exp(|i-I|+1)
\\
\mathbf{q}_s = \mathbf{W}_c[\mathbf{x}_{s,I},\mathbf{x}^*_s,\mathbf{p}_s]
$$
$\mathbf{x}_{s,I}$表示最后一次点击，$\mathbf{x^*_s}$表示聚合后的会话表示，$\mathbf{p}_s$表示加入物品相对位置信息的会话表示，$\mathbf{q}_s $是最后的会话表示。

3）loss = $\mathit{L_in}$

#### 2.对全局转换动态建模

1）用图神经网络结构和GCN对inter-session的依赖建模

2）用**互信息学习**来增强跨会话的建模物品间关系的encoder

3）loss = $\mathit{L_cr}$

#### 3.Model Inference

定义loss：
$$
\mathit{L}=\mathit{L}_{cr} + \lambda_1\mathit{L}_{in}+\lambda_2||\Theta||^2_2
$$
