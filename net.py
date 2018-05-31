# -*- coding: utf-8 -*-
import sys

import math
import random
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim
from conf import *

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
class Network(nn.Module):
    def __init__(self, embedding_size, embedding_dimention, embedding_matrix, hidden_dimention, output_dimention):
        super(Network,self).__init__()
        self.embedding_layer = nn.Embedding(embedding_size,embedding_dimention)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.inpt_layer_zp_pre = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_zp_pre = nn.Linear(hidden_dimention,hidden_dimention)
        self.inpt_layer_zp_post = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_zp_post = nn.Linear(hidden_dimention,hidden_dimention)
        self.inpt_layer_np = nn.Linear(embedding_dimention,hidden_dimention)
        self.hidden_layer_np = nn.Linear(hidden_dimention,hidden_dimention)
        self.inpt_layer_nps = nn.Linear(hidden_dimention,hidden_dimention)
        self.hidden_layer_nps = nn.Linear(hidden_dimention,hidden_dimention)
        nh = hidden_dimention*2
        self.zp_pre_layer = nn.Linear(hidden_dimention,nh)
        self.zp_post_layer = nn.Linear(hidden_dimention,nh)
        self.np_layer = nn.Linear(hidden_dimention,nh)
        self.nps_layer = nn.Linear(hidden_dimention*2,nh)
        self.feature_layer = nn.Linear(61,nh)
        self.representation_hidden_layer = nn.Linear(hidden_dimention*2,hidden_dimention*2)
        self.output_layer = nn.Linear(hidden_dimention*2,output_dimention)
        self.hidden_size = hidden_dimention
        self.activate = nn.Tanh()
        self.softmax_layer = nn.Softmax()
    def forward_zp_pre(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)
        word_embedding = dropout_layer(word_embedding)
        this_hidden = self.inpt_layer_zp_pre(word_embedding) + self.hidden_layer_zp_pre(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden
    def forward_zp_post(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)
        this_hidden = self.inpt_layer_zp_post(word_embedding) + self.hidden_layer_zp_post(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden
    def forward_np(self, word_index, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        word_embedding = self.embedding_layer(word_index)
        this_hidden = self.inpt_layer_np(word_embedding) + self.hidden_layer_np(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden
    def forward_nps(self, inpt, hiden_layer,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        this_hidden = self.inpt_layer_nps(inpt) + self.hidden_layer_np(hiden_layer)
        this_hidden = self.activate(this_hidden)
        this_hidden = dropout_layer(this_hidden)
        return this_hidden
    def generate_score(self,zp_pre,zp_post,np,feature,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        x = self.zp_pre_layer(zp_pre) + self.zp_post_layer(zp_post) + self.np_layer(np)\
            + self.feature_layer(feature) 
        x = self.activate(x)
        x = dropout_layer(x)
        x = self.representation_hidden_layer(x)
        x = self.activate(x)
        x = dropout_layer(x)
        x = self.output_layer(x)
        xs = F.softmax(x)
        return x,xs
    def generate_scores(self,zp_pre,zp_post,np,nps,feature,dropout=0.0):
        dropout_layer = nn.Dropout(dropout)
        x = self.zp_pre_layer(zp_pre) + self.zp_post_layer(zp_post) + self.np_layer(np) + self.nps_layer(nps)\
            + self.feature_layer(feature) 
        x = self.activate(x)
        x = dropout_layer(x)
        x = self.representation_hidden_layer(x)
        x = self.activate(x)
        x = dropout_layer(x)
        x = self.output_layer(x)
        xs = F.softmax(x)
        return x,xs
    def initHidden(self,batch=1):
        return autograd.Variable(torch.from_numpy(numpy.zeros((batch, self.hidden_size))).type(torch.cuda.FloatTensor))
