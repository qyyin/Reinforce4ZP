#coding=utf8
import argparse
import random
import numpy
import properties_loader
import sys
import torch
import collections
from utils import *

DIR="./data/"
parser = argparse.ArgumentParser(description="Experiemts\n")
parser.add_argument("-data",default = DIR, type=str, help="saved vectorized data")
parser.add_argument("-raw_data",default = "./data/zp_data/", type=str, help="raw_data")
parser.add_argument("-props",default = "./properties/prob", type=str, help="properties")
parser.add_argument("-reduced",default = 0, type=int, help="reduced")
parser.add_argument("-gpu",default = 0, type=int, help="GPU number")
parser.add_argument("-random_seed",default=0,type=int,help="random seed")
parser.add_argument("-round",default=70,type=int,help="random seed")
args = parser.parse_args()

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
nnargs = properties_loader.read_pros(args.props)
