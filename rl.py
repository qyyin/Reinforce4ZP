#coding=utf8
import os
import sys
import re
import math
import timeit
import cPickle
import copy
sys.setrecursionlimit(1000000)
import torch
import torch.nn as nn
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
    # reinforcement learning
    if os.path.isfile("./data/train_data"):
        read_f = file("./data/train_data","rb")
        train_generater = cPickle.load(read_f)
        read_f.close()
    else:
        train_generater = DataGnerater("train",nnargs["batch_size"])
        train_generater.devide()
    test_generater = DataGnerater("test",256)

    read_f = file("./data/emb","rb")
    embedding_matrix,_,_ = cPickle.load(read_f)
    read_f.close()

    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2).cuda()
    model_ = torch.load("./models/model")
    mcp = list(model.parameters())
    mp = list(model_.parameters())
    n = len(mcp)
    for i in range(0, n): 
        mcp[i].data[:] = mp[i].data[:]
    optimizer = optim.Adagrad(model.parameters(),lr=nnargs["rl"])
    best = {"sum":0.0}
    best_model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],2).cuda()
    re = evaluate_test(test_generater,model)
    print "Performance on Test Before RL: F",re["f"]
    for echo in range(50):
        info = "["+echo*">"+" "*(50-echo)+"]"
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
            output,output_softmax = model.generate_score(zp_pre_rep,zp_post_rep,candi_rep,feature,dropout=nnargs["dropout"])
            target = autograd.Variable(torch.from_numpy(data["result"]).type(torch.cuda.LongTensor))
            nps = torch.zeros(len(candi_rep),len(candi_rep)).type(torch.cuda.FloatTensor)
            for s,e in data["s2e"]:
                if s == e:continue
                thre = output_softmax[s:e][:,1].data.cpu().numpy()
                lu = numpy.clip(numpy.floor(numpy.random.rand(len(thre)) / thre), 1, 0).astype(int)
                heihei = torch.from_numpy(lu).type(torch.cuda.FloatTensor)
                for i in range(1,len(lu)):
                    nps[s+i][s:s+i] = heihei[:i]
            nps = autograd.Variable(nps)
            history = nps.view(len(candi_rep),len(candi_rep),1)*candi_rep
            maxh,_ = torch.max(history,1)
            ave = torch.sum(history,1)/(torch.sum(nps.view(len(candi_rep),len(candi_rep),1),1)+1e-10)
            history = torch.cat([maxh,ave],1)
            _,output_softmax = model.generate_scores(zp_pre_rep,zp_post_rep,candi_rep,history,feature,dropout=nnargs["dropout"])
            thre = output_softmax[:,1].data.cpu().numpy()
            lu = numpy.clip(numpy.floor(numpy.random.rand(len(thre)) / thre), 1, 0).astype(int)
            gold = data["target"]
            if float(sum(gold)) == 0 or sum(gold*lu) == 0 or sum(lu) == 0:continue
            prec = float(sum(gold*lu))/float(numpy.count_nonzero(lu))
            rec = float(sum(gold*lu))/float(numpy.count_nonzero(gold))
            sc = 0 if (rec == 0.0 or prec == 0.0) else 2.0/(1.0/prec+1.0/rec)
            if sc == 0:continue
            rewards = numpy.full((len(lu),2),sc)
            pl = lu.tolist()
            for i in range(len(pl)):
                np = copy.deepcopy(pl)
                np[i] = 1-np[i]
                if float(sum(gold)) == 0 or sum(gold*np) == 0 or sum(np) == 0:
                    nsc = 0.0
                else:
                    nprec = float(sum(gold*np))/float(numpy.count_nonzero(np))
                    nrec = float(sum(gold*np))/float(numpy.count_nonzero(gold))
                    nsc = 0.0 if (nrec == 0.0 or nprec == 0.0) else 2.0/(1.0/nprec+1.0/nrec)
                rewards[i][np[i]] = nsc
            maxs = rewards.min(axis=1)[:,numpy.newaxis]
            rewards = rewards - maxs
            rewards = torch.tensor(-1.0*rewards).type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            loss = torch.sum(output_softmax*rewards) 
            loss.backward()
            optimizer.step()
        re = evaluate(train_generater,model)
        if re >= best["sum"]:
            mcp = list(best_model.parameters())
            mp = list(model.parameters())
            for i in range(0, len(mcp)): 
                mcp[i].data[:] = mp[i].data[:]
            best["sum"] = re

    print >> sys.stderr
    re = evaluate_test(test_generater,best_model)
    print "Performance on Test Final: F",re["f"]
    torch.save(best_model, "./models/model.final")
    print "Dev",best["sum"]

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
        nps = torch.zeros(len(candi_rep),len(candi_rep)).type(torch.cuda.FloatTensor)
        for s,e in data["s2e"]:
            if s == e:
                continue
            thre = output_softmax[s:e][:,1].data.cpu().numpy()
            lu = numpy.clip(numpy.floor(0.5 / thre), 1, 0).astype(int)
            heihei = torch.tensor(lu).type(torch.cuda.FloatTensor)
            for i in range(1,len(lu)):
                nps[s+i][s:s+i] = heihei[:i]
        history = nps.view(len(candi_rep),len(candi_rep),1)*candi_rep
        maxh,_ = torch.max(history,1)
        ave = torch.sum(history,1)/(torch.sum(nps.view(len(candi_rep),len(candi_rep),1),1)+1e-10)
        history = torch.cat([maxh,ave],1)
        output,output_softmax = model.generate_scores(zp_pre_rep,zp_post_rep,candi_rep,history,feature)
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
    return sum(predict)/float(len(predict))

def evaluate_test(generater,model):
    pr = []
    for data in generater.generate_data():
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
        nps = torch.zeros(len(candi_rep),len(candi_rep)).type(torch.cuda.FloatTensor)
        for s,e in data["s2e"]:
            if s == e:
                continue
            thre = output_softmax[s:e][:,1].data.cpu().numpy()
            lu = numpy.clip(numpy.floor(0.5 / thre), 1, 0).astype(int)
            heihei = torch.tensor(lu).type(torch.cuda.FloatTensor)
            for i in range(1,len(lu)):
                nps[s+i][s:s+i] = heihei[:i]
        history = nps.view(len(candi_rep),len(candi_rep),1)*candi_rep
        maxh,_ = torch.max(history,1)
        ave = torch.sum(history,1)/(torch.sum(nps.view(len(candi_rep),len(candi_rep),1),1)+1e-10)
        history = torch.cat([maxh,ave],1)
        output,output_softmax = model.generate_scores(zp_pre_rep,zp_post_rep,candi_rep,history,feature)
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
    p = sum(predict)/float(len(predict))
    r = sum(predict)/1713.0
    f = 0.0 if (p == 0 or r == 0) else (2.0/(1.0/p+1.0/r))
    re = {"p":p,"r":r,"f":f}
    return re


if __name__ == "__main__":
    main()
