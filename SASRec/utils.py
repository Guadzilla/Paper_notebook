import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)  # 随机采样user id，注意是从1开始的
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)  # 长度小于1的训练集不要

        seq = np.zeros([maxlen], dtype=np.int32)  # seq序列，长度固定为maxlen，用0在前面padding补上长度，例：[0,0,...,0,23,15,2,6]
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]  # user_train的最后一个item取为nxt
        idx = maxlen - 1

        ts = set(user_train[user])  # ts为序列的item集合
        for i in reversed(user_train[user][:-1]):  # 从后往前遍历user_train,idx为当前要填充的下标
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  # 生成的负样本不能取该序列item集合里的item
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)  # 返回一次采样，(用户id,训练序列，label序列，负样本序列)

    np.random.seed(SEED)
    while True:  # 采样一个batch_size大小的数据样本，打包成一个batch，放到线程队列里
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)  # 长度为10的线程队列
        self.processors = []
        for i in range(n_workers):
            self.processors.append(  # Process()进程的类, target：要调用的对象即sampler_function，args：调用该对象要接受的参数
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


# train/val/test data generation
# 生成训练集/验证集/测试集
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)  # 字典默认value为list类型
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)  # user和item总数就是各自数值最大的那个，应该跟预处理有关
        itemnum = max(i, itemnum)
        User[u].append(i)  # User[1] = [1,2,3,4,5] => 用户1依次点击1,2,3,4,5

    for user in User:
        nfeedback = len(User[user])  # nfeedback，每个用户与物品的交互序列长度
        if nfeedback < 3:  # 如果序列长度小于3，不够划分验证集和测试集
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:  # 序列够划分的话，前n-2个交互作为训练集，第n-1个item作为验证label，第n个作为测试label
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]  # 返回[训练集,验证label,测试label,用户总数,物品总数]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)  # deepcopy一份用于valid和test

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:  # 用户数量大于10000就随机采10000
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        # 假设原始序列为[1,2,3,4,5,6,7]    6用于valid；7用于test
        seq[idx] = valid[u][0]  # seq: [0,0,0,...,0,0,0,6]
        idx -= 1
        for i in reversed(train[u]):  # seq: [0,0,0,...,0,1,2,3,4,5,6]  只剩test里的[7]用于预测
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])  # 序列物品集合
        rated.add(0)
        item_idx = [test[u][0]]  # 取出ground truth label
        for _ in range(100):  # item_idx: [label,random,random,...,random] 1+100个随机物品，看得分在top10的情况
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # (1,101) -> 101 (squeeze)

        rank = predictions.argsort().argsort()[0].item()  # 做两次argsort()，可以得到每个位置的rank排名

        valid_user += 1

        if rank < 10:  # TOP10才记录，这里真实rank = rank + 1 ，因为argsort()索引包含0
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            # sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]    # 取出valid label
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
