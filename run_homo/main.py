import argparse

from model import LogReg, Model
from utils import load

import torch
import torch as th
import torch.nn as nn
import  numpy as np
import warnings
import random
warnings.filterwarnings('ignore')
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)
parser = argparse.ArgumentParser(description='SAGCL')

parser.add_argument('--dataname', type=str, default='comp', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=5e-4, help='Learning rate of pretraining.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=1e-6, help='Weight decay of pretraining.')
parser.add_argument('--wd2', type=float, default=1e-5, help='Weight decay of linear evaluator.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature hyperparameter.')
parser.add_argument("--hid_dim", type=int, default=2048, help='Hidden layer dim.')
parser.add_argument('--moving_average_decay', type=float, default=0.9)
parser.add_argument('--num_MLP', type=int, default=1)
parser.add_argument('--lambda_loss', type=float, default=1)
parser.add_argument('--run_times', type=int, default=1)
args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

# def write_results(acc):
#     f = open("results/" + args.dataname + '_heterSSL', 'a+')
#     f.write(args.dataname + ' --lr1' + str(args.lr1) + ' --lambda_loss ' + str(
#         args.lambda_loss) + ' --moving_average_decay ' + str(args.moving_average_decay) + ' --dimension ' + str(
#         args.hid_dim) + ' --tau ' + str(args.temp) + ' --num_MLP ' + str(args.num_MLP) + f'   Final Test: {np.mean(acc):.4f} ± {np.std(acc):.4f}\n')
#     f.close()

if __name__ == '__main__':
    print(args)
    # load hyperparameters
    dataname = args.dataname
    hid_dim = args.hid_dim
    out_dim = args.hid_dim
    n_layers = args.n_layers
    temp = args.temp
    epochs = args.epochs
    lr1 = args.lr1
    wd1 = args.wd1
    lr2 = args.lr2
    wd2 = args.wd2
    device = args.device

    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(dataname)
    in_dim = feat.shape[1]

    model = Model(in_dim, hid_dim, out_dim, n_layers, temp, args.use_mlp, args.moving_average_decay, args.num_MLP, args.lambda_loss)
    model = model.to(device)

    graph = graph.to(device)
    feat = feat.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph = graph.remove_self_loop().add_self_loop()
        loss = model(graph, feat)
        loss.backward()
        optimizer.step()
        model.update_moving_average()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    print("=== Evaluation ===")
    graph = graph.remove_self_loop().add_self_loop()
    embeds = model.get_embedding(graph, feat)
    results = []
    for run in range(0, args.run_times):

        train_idx_tmp = train_idx
        val_idx_tmp = val_idx
        test_idx_tmp = test_idx
        train_embs = embeds[train_idx_tmp]
        val_embs = embeds[val_idx_tmp]
        test_embs = embeds[test_idx_tmp]

        label = labels.to(device)

        train_labels = label[train_idx_tmp]
        val_labels = label[val_idx_tmp]
        test_labels = label[test_idx_tmp]

        train_feat = feat[train_idx_tmp]
        val_feat = feat[val_idx_tmp]
        test_feat = feat[test_idx_tmp]

        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], num_class)
        opt = th.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

        logreg = logreg.to(device)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0
        eval_acc = 0

        for epoch in range(1000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc

                    eval_acc = test_acc

                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
        results.append(eval_acc.item())
        print(f'Validation Accuracy: {best_val_acc}, Test Accuracy: {eval_acc}')
    # write_results(results)


