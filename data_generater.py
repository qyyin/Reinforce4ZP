#coding=utf8
import os
import sys
import re
import math
import timeit
from subprocess import *
from conf import *
import cPickle
import collections
sys.setrecursionlimit(1000000)

class DataGnerater():
    def __init__(self,file_type,max_pair):
        data_path = args.data+file_type+"/" 
        if args.reduced == 1:
            data_path = args.data+file_type + "_reduced/"
        self.candi_vec = numpy.load(data_path+"candi_vec.npy")
        self.candi_vec_mask = numpy.load(data_path+"candi_vec_mask.npy")
        self.ifl_vec = numpy.load(data_path+"ifl_vec.npy")
        self.zp_post = numpy.load(data_path+"zp_post.npy")
        self.zp_post_vec_mask = numpy.load(data_path+"zp_post_mask.npy")
        self.zp_pre = numpy.load(data_path+"zp_pre.npy")
        self.zp_pre_vec_mask = numpy.load(data_path+"zp_pre_mask.npy")

        read_f = file(data_path + "zp_candi_pair_info","rb")
        zp_candis_pair = cPickle.load(read_f)
        read_f.close()
        self.data_batch = []
        zp_rein = []
        candi_rein = []
        this_target = [] 
        this_result = []
        s2e = []
        for i in range(len(zp_candis_pair)):
            zpi,candis = zp_candis_pair[i]
            if len(candis)+len(candi_rein) > max_pair and len(candi_rein) > 0:
                ci_s = candi_rein[0]
                ci_e = candi_rein[-1]+1
                zpi_s = zp_rein[0]
                zpi_e = zp_rein[-1]+1
                this_batch = {}
                this_batch["zp_rein"] = numpy.array(zp_rein,dtype="int32")-zp_rein[0]
                this_batch["candi_rein"] = numpy.array(candi_rein,dtype="int32")-candi_rein[0]
                this_batch["target"] = numpy.array(this_target,dtype="int32")
                this_batch["result"] = numpy.array(this_result,dtype="int32")
                this_batch["zp_post"] = self.zp_post[zpi_s:zpi_e]
                this_batch["zp_pre"] = self.zp_pre[zpi_s:zpi_e]
                this_batch["zp_post_mask"] = self.zp_post_vec_mask[zpi_s:zpi_e]
                this_batch["zp_pre_mask"] = self.zp_pre_vec_mask[zpi_s:zpi_e]
                this_batch["candi"] = self.candi_vec[ci_s:ci_e]
                this_batch["candi_mask"] = self.candi_vec_mask[ci_s:ci_e]
                this_batch["fl"] = self.ifl_vec[ci_s:ci_e]
                this_batch["s2e"] = s2e
                self.data_batch.append(this_batch)
                zp_rein = []
                candi_rein = []
                this_target = [] 
                this_result = []
                s2e = []
            start = len(this_result)
            end = start
            for candii,res,tar in candis:
                zp_rein.append(zpi)
                candi_rein.append(candii)
                this_target.append(tar)
                this_result.append(res)
                end += 1
            s2e.append((start,end))
        if len(candi_rein) > 0:
            ci_s = candi_rein[0]
            ci_e = candi_rein[-1]+1
            zpi_s = zp_rein[0]
            zpi_e = zp_rein[-1]+1
            this_batch = {}
            this_batch["zp_rein"] = numpy.array(zp_rein,dtype="int32")-zp_rein[0]
            this_batch["candi_rein"] = numpy.array(candi_rein,dtype="int32")-candi_rein[0]
            this_batch["target"] = numpy.array(this_target,dtype="int32")
            this_batch["result"] = numpy.array(this_result,dtype="int32")
            this_batch["zp_post"] = self.zp_post[zpi_s:zpi_e]
            this_batch["zp_pre"] = self.zp_pre[zpi_s:zpi_e]
            this_batch["zp_post_mask"] = self.zp_post_vec_mask[zpi_s:zpi_e]
            this_batch["zp_pre_mask"] = self.zp_pre_vec_mask[zpi_s:zpi_e]
            this_batch["candi"] = self.candi_vec[ci_s:ci_e]
            this_batch["candi_mask"] = self.candi_vec_mask[ci_s:ci_e]
            this_batch["fl"] = self.ifl_vec[ci_s:ci_e]
            this_batch["s2e"] = s2e
            self.data_batch.append(this_batch)

    def devide(self,k=0.2):
        random.shuffle(self.data_batch)
        length = int(len(self.data_batch)*k)
        self.dev = self.data_batch[:length]
        self.train = self.data_batch[length:]
        self.data_batch = self.train

    def generate_data(self,shuffle=False):
        if shuffle:
            random.shuffle(self.data_batch) 
        for data in self.data_batch:
            yield data

    def generate_dev_data(self,shuffle=False):
        if shuffle:
            random.shuffle(self.dev) 
        for data in self.dev:
            yield data
