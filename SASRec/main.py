import os
import time
import torch
import argparse

from model import SASRec
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


import lazy_cuda

# 需要的参数，可以通过命令行指定
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default=lazy_cuda.select_device(), type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()
# 创建保存模型的文件夹
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    # 保存arg指定的超参数
    # vars(object)返回对象object的属性和属性值的字典对象
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)  # data_partition 返回 [训练集,验证label,测试label,用户总数,物品总数]

    [user_train, user_valid, user_test, usernum, itemnum] = dataset  # 从dataset里取出来
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))  # 计算序列平均长度

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')  # 新建一个日志，用户后续保存结果

    # sampler是WarpSampler对象的实例，每次调用sampler.next_batch(),就返回一个batch的样本。
    # 进一步解释：每次调用sampler.next_batch()就call其线程队列里的一个线程，每个线程用于返回一个batch的数据。
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():  # xavier初始化参数
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:  # 读取保存好的模型，保存在 state_dict_path
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            # u: batch_size
            # seq, pos, neg: batch_size * maxlen
            # (用户id,训练序列，positive label序列，negative label序列)
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), \
                                     torch.zeros(neg_logits.shape, device=args.device)


            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits)
            # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)    # 返回一个二维数组array， array[0]=[横坐标], array[1]=[纵坐标]
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])  # 正例的loss
            loss += bce_criterion(neg_logits[indices], neg_labels[indices]) # 负例的loss
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)   # l2正则化，但参数默认为0
            loss.backward()
            adam_optimizer.step()
            if step % (num_batch//5) == 0:
                print("epoch: {0} ,iteration: {1:3d} ,loss: {2:.5f}".format(epoch, step, loss.item()))  # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0 or epoch == args.num_epochs:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('\nepoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)\n'
                  % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write('valid:' + str(t_valid) + '; test:' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            if epoch == args.num_epochs:
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                     args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))     # 训练完成就保存模型
            else:
                model.train()



    f.close()
    sampler.close()     # 关闭线程
    print("Done")
