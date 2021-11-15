# SASRec代码笔记

## collections.defaultdict(list)

```python
from collections import defaultdict
result = defaultdict(list)
data = [("p", 1), ("p", 2), ("p", 3),
        ("h", 1), ("h", 2), ("h", 3)]
 
for (key, value) in data:
    result[key].append(value)
    
print(result)#defaultdict(<class 'list'>, {'p': [1, 2, 3], 'h': [1, 2, 3]})

```



## 手动多线程

```python
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)    # 随机采样user id，注意是从1开始的
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)  # 长度小于1的训练集不要

        seq = np.zeros([maxlen], dtype=np.int32)    # seq序列，长度固定为maxlen，用0在前面padding补上长度，例：[0,0,...,0,23,15,2,6]
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]  # user_train的最后一个item取为nxt
        idx = maxlen - 1

        ts = set(user_train[user])  # ts为序列的item集合
        for i in reversed(user_train[user][:-1]):   # 从后往前遍历user_train,idx为当前要填充的下标
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  # 生成的负样本不能取该序列item集合里的item
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)    # 返回一次采样，(用户id,训练序列，label序列，负样本序列)

    np.random.seed(SEED)
    while True:     # 采样一个batch_size大小的数据样本，打包成一个batch，放到线程队列里
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)   # 长度为10的线程队列
        self.processors = []
        for i in range(n_workers):
            self.processors.append(     # Process()进程的类, target：要调用的对象即sampler_function，args：调用该对象要接受的参数
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
            
            
            
# sampler是WarpSampler对象的实例，每次调用sampler.next_batch(),就返回一个batch的样本。
# 进一步解释：每次调用sampler.next_batch()就call其线程队列里的一个线程，每个线程用于返回一个batch的数据。
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
```



## torch.tril()

```python
torch.tril(input, diagonal=0, *, out=None) → Tensor
# 功能：返回下三角矩阵其余部分用out填充（默认为0）
# input：输入矩阵，二维tensor
# diagonal：表示对角线位置，diagonal=0为主对角线，diagonal=-1为主对角线往下1格，diagonal=1为主对角线往上1格
# out：表示填充，默认用out=None即0填充
```

例：

```python
>>> a = torch.randn(3, 3)
>>> a
tensor([[-1.0813, -0.8619,  0.7105],
        [ 0.0935,  0.1380,  2.2112],
        [-0.3409, -0.9828,  0.0289]])
>>> torch.tril(a)
tensor([[-1.0813,  0.0000,  0.0000],
        [ 0.0935,  0.1380,  0.0000],
        [-0.3409, -0.9828,  0.0289]])

>>> b = torch.randn(4, 6)
>>> b
tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
        [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
        [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
        [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
>>> torch.tril(b, diagonal=1)
tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
        [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
        [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
>>> torch.tril(b, diagonal=-1)
tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])
```



## python中的 ~ 波浪线运算符

~，用法只有一个那就是按位取反

[ Python 波浪线与补码_https://space.bilibili.com/59807853-CSDN博客_python 波浪线](https://blog.csdn.net/lanchunhui/article/details/51746477)



## torch.nn.MultiAttention

```python
torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):

```

对应公式：
$$
Multihead(Q,K,V) = Concat(head_1,...,head_h)W^O	\\
where \quad head_i= Attention(QW^Q_i,KW^K_i,VW^V_i)
$$


计算公式：

```python
forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None)
```

QKV比较常规，需要注意的是

1. key_padding_mask参数，大小为（N，S），指定key中的哪些元素不做attention计算，即看作padding。注意，为True的位置不计算attention（是padding的地方不计算）
2. attn_mask参数，



## torch.nn.BCEWithLogitsLoss()

```
forward(self, input: Tensor, target: Tensor) -> Tensor
```

参数说明：

- input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
- target: Tensor of the same shape as input with values between 0 and 1

input：$x$        output：$y$



$ℓ(x,y)=L={l_1,…,l_N}^T\\l_n=−w_n[y_n·log\sigma(x_n)+(1−y_n)·log(1−\sigma(x_n))]$



当 $y=1$ 时，$l_n=−log\sigma(x_n)$  ；当 $y=0$ 时，$l_n=−log(1-\sigma(x_n))$ 	。

论文里使用了一个全1的矩阵pos_labels，和一个全0的矩阵neg_labels。正例标签值都为1（正确的item，ground truth应该是概率为1），负例标签值都为0（错误的item，ground truth应该是概率为0）。

```python
pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), \
torch.zeros(neg_logits.shape, device=args.device)


# print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits)
# check pos_logits > 0, neg_logits < 0
adam_optimizer.zero_grad()
indices = np.where(pos != 0)    # 返回一个二维数组array， array[0]=[横坐标], array[1]=[纵坐标]
loss = bce_criterion(pos_logits[indices], pos_labels[indices])  # 使正例的得分尽量
loss += bce_criterion(neg_logits[indices], neg_labels[indices])
```



## torch.argsort()

```python
torch.argsort(input, dim=-1, descending=False) → LongTensor
```

沿着指定dim从小到大（默认）排序元素，然后返回这些元素原来的下标。

```python
>>>t = torch.randint(1,10,(1,5))
>>>t
tensor([[7, 9, 5, 6, 3]])
>>>t.argsort()
tensor([[4, 2, 3, 0, 1]])
>>>t.argsort().argsort()
tensor([[3, 4, 1, 2, 0]])

# 两次argsort()可以返回每个元素的rank排名
# 解释：
# 把商品0,1,2,3,4按顺序摆好，他们的得分分别为[7,9,5,6,3]
# 对所有商品的得分从小到大排序（argsort()操作）
# 得到积分排名是[3,5,6,7,9]，积分排名对应的商品id是[4,2,3,0,1]（第一次argsort()的结果），每个商品id对应的下标就是他们的得分名次
# 例如商品4得分最高排在第一位，商品1得分最低排最后一位
# 然后我们想得到0,1,2,3,4顺序下的结果
# 所以对商品id排序，使得商品摆放顺序由[4,2,3,0,1]变为[0,1,2,3,4]，这里也是argsort()操作，因为0~4天然有顺序关系
# [4,2,3,0,1]变为[0,1,2,3,4]的同时，排名情况[0,1,2,3,4]也变成了[3,4,1,2,0]（第二次argsort()的结果）
# 即求得每个商品在原来顺序下的得分名次
```

[numpy中的argmax、argmin、argwhere、argsort、argpartition函数 - 古明地盆 - 博客园 (cnblogs.com)](https://www.cnblogs.com/traditional/p/13702904.html)

## 评价指标Hit Ratio、NDCG[1]

### Hit Ratio

Evaluation Metrics. Given a user, each algorithm produces a ranked list of items. To assess the ranked list with the ground-truth item set (GT), we adopt Hit Ratio (HR), which has been commonly used in top-N evaluation . If a test item appears in the recommended list, it is deemed a hit. HR is calculated as:
$$
HR@K=\frac{Number\ of \  Hits@K}{|GT|}
$$

### NDCG

As the HR is recall-based metric, it does not reflect the accuracy of getting top ranks correct, which is crucial in many real-world applications. To address this, we also adopt Normalized Discounted Cumulative Gain (NDCG), which assigns higher importance to results at top ranks, scoring successively lower ranks with marginal fractional utility:
$$
NDCG@K=Z_K\sum^K_{i=1}\frac{2^{r_i}-1}{log_2{(i+1)}}
$$
where ZK is the normalizer to ensure the perfect ranking has a value of 1; ri is the graded relevance of item at position i. We use the simple binary relevance for our work: ri = 1 if the item is in the test set, and 0 otherwise. For both metrics, larger values indicate better performance. In the evaluation, we calculate both metrics for each user in the test set, and report the average score.



## 参考文献：

[1]He X, Chen T, Kan M Y, et al. Trirank: Review-aware explainable recommendation by modeling aspects[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. 2015: 1661-1670.



