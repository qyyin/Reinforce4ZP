#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import cPickle
import copy
sys.setrecursionlimit(1000000)
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import lr_scheduler
from conf import *
from data_generater import *
from net import *

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
print "PID", os.getpid()
torch.cuda.set_device(args.gpu)

def main():
    # pretraining file
    read_f = file("./data/train_data","rb")
    train_generater = cPickle.load(read_f)
    read_f.close()
    read_f = file("./data/emb","rb")
    embedding_matrix,_,_ = cPickle.load(read_f)
    read_f.close()

    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2).cuda()
    best_model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2).cuda()
    this_lr = 0.003 
    optimizer = optim.Adagrad(model.parameters(), lr=this_lr)
    best = {"sum":0.0}
    print "Pretrain"
    for echo in range(args.round):
        info = "["+echo*">"+" "*(args.round-echo)+"]"
        sys.stderr.write(info+"\r")
        for data in train_generater.generate_data(shuffle=True):
            zp_rein = torch.tensor(data["zp_rein"]).type(torch.cuda.LongTensor)
            zp_pre = torch.tensor(data["zp_pre"]).type(torch.cuda.LongTensor)
            zp_pre_mask = torch.tensor(data["zp_pre_mask"]).type(torch.cuda.FloatTensor)
            zp_post = torch.tensor(data["zp_post"]).type(torch.cuda.LongTensor)
            zp_post_mask = torch.tensor(data["zp_post_mask"]).type(torch.cuda.FloatTensor)
            candi_rein = torch.tensor(data["candi_rein"]).type(torch.cuda.LongTensor)
            candi = torch.tensor(data["candi"]).type(torch.cuda.LongTensor)
            candi_mask = torch.tensor(data["candi_mask"]).type(torch.cuda.FloatTensor)
            feature = torch.tensor(data["fl"]).type(torch.cuda.FloatTensor)
            zp_pre = torch.transpose(zp_pre,0,1)
            mask_zp_pre = torch.transpose(zp_pre_mask,0,1)
            hidden_zp_pre = model.initHidden()
            for i in range(len(mask_zp_pre)):
                hidden_zp_pre = model.forward_zp_pre(zp_pre[i],hidden_zp_pre,dropout=nnargs["dropout"])*torch.transpose(mask_zp_pre[i:i+1],0,1)
            zp_pre_rep = hidden_zp_pre[zp_rein]
            zp_post = torch.transpose(zp_post,0,1)
            mask_zp_post = torch.transpose(zp_post_mask,0,1)
            hidden_zp_post = model.initHidden()
            for i in range(len(mask_zp_post)):
                hidden_zp_post = model.forward_zp_post(zp_post[i],hidden_zp_post,dropout=nnargs["dropout"])*torch.transpose(mask_zp_post[i:i+1],0,1)
            zp_post_rep = hidden_zp_post[zp_rein]
            candi = torch.transpose(candi,0,1)
            mask_candi = torch.transpose(candi_mask,0,1)
            hidden_candi = model.initHidden()
            for i in range(len(mask_candi)):
                hidden_candi = model.forward_np(candi[i],hidden_candi,dropout=nnargs["dropout"])*torch.transpose(mask_candi[i:i+1],0,1)
            candi_rep = hidden_candi[candi_rein]
            assert len(feature) == len(candi_rep)
            assert len(zp_post_rep) == len(candi_rep)
            output,output_softmax = model.generate_score(zp_pre_rep,zp_post_rep,candi_rep,feature,dropout=nnargs["dropout"])
            optimizer.zero_grad()
            loss = F.cross_entropy(output,torch.tensor(data["result"]).type(torch.cuda.LongTensor))
            loss.backward()
            optimizer.step()
        re = evaluate(train_generater,model)
        if re > best["sum"]:
            best["model"] = model
            best["sum"] = re
    print >> sys.stderr
    best_model = best["model"]
    torch.save(best_model, "./models/model") 

def evaluate(generater,model):
    pr = []
    for data in generater.generate_dev_data():
        zp_rein = torch.tensor(data["zp_rein"]).type(torch.cuda.LongTensor)
        zp_pre = torch.tensor(data["zp_pre"]).type(torch.cuda.LongTensor)
        zp_pre_mask = torch.tensor(data["zp_pre_mask"]).type(torch.cuda.FloatTensor)
        zp_post = torch.tensor(data["zp_post"]).type(torch.cuda.LongTensor)
        zp_post_mask = torch.tensor(data["zp_post_mask"]).type(torch.cuda.FloatTensor)
        candi_rein = torch.tensor(data["candi_rein"]).type(torch.cuda.LongTensor)
        candi = torch.tensor(data["candi"]).type(torch.cuda.LongTensor)
        candi_mask = torch.tensor(data["candi_mask"]).type(torch.cuda.FloatTensor)
        feature = torch.tensor(data["fl"]).type(torch.cuda.FloatTensor)
        zp_pre = torch.transpose(zp_pre,0,1)
        mask_zp_pre = torch.transpose(zp_pre_mask,0,1)
        hidden_zp_pre = model.initHidden()
        for i in range(len(mask_zp_pre)):
            hidden_zp_pre = model.forward_zp_pre(zp_pre[i],hidden_zp_pre)*torch.transpose(mask_zp_pre[i:i+1],0,1)
        zp_pre_rep = hidden_zp_pre[zp_rein]
        zp_post = torch.transpose(zp_post,0,1)
        mask_zp_post = torch.transpose(zp_post_mask,0,1)
        hidden_zp_post = model.initHidden()
        for i in range(len(mask_zp_post)):
            hidden_zp_post = model.forward_zp_post(zp_post[i],hidden_zp_post)*torch.transpose(mask_zp_post[i:i+1],0,1)
        zp_post_rep = hidden_zp_post[zp_rein]
        candi = torch.transpose(candi,0,1)
        mask_candi = torch.transpose(candi_mask,0,1)
        hidden_candi = model.initHidden()
        for i in range(len(mask_candi)):
            hidden_candi = model.forward_np(candi[i],hidden_candi)*torch.transpose(mask_candi[i:i+1],0,1)
        candi_rep = hidden_candi[candi_rein]
        output,output_softmax = model.generate_score(zp_pre_rep,zp_post_rep,candi_rep,feature)
        output_softmax = output_softmax.data.cpu().numpy()
        for s,e in data["s2e"]:
            if s == e:
                continue
            pr.append((data["result"][s:e],output_softmax[s:e]))
    predict = []
    for result,output in pr:
        index = -1
        pro = 0.0
        for i in range(len(output)):
            if output[i][1] > pro:
                index = i
                pro = output[i][1]
        predict.append(result[index])
    return sum(predict)

if __name__ == "__main__":
    main()
