#coding=utf8
import os
import sys
import re
import parse_analysis
from subprocess import *

def dif(l1,l2):
    if not (len(l1) == len(l2)):
        return True
    for i in range(len(l1)):
        if not (l1[i] == l2[i]):
            return True
    return False

def is_pro(leaf_nodes):
    if len(leaf_nodes) == 1:
        if leaf_nodes[0].word == "*pro*":
            return True
    return False

def is_zero_tag(leaf_nodes):
    if len(leaf_nodes) == 1:
        if leaf_nodes[0].word.find("*") >= 0:
            return True
    return False

def is_np(tag):
    np_list = ['NP-SBJ', 'NP', 'NP-PN-OBJ', 'NP-PN', 'NP-PN-SBJ', 'NP-OBJ', 'NP-TPC-1', 'NP-TPC', 'NP-PN-VOC', 'NP-VOC', 'NP-IO', 'NP-SBJ-1', 'NP-PN-TPC', 'NP-PRD', 'NP-TMP', 'NP-PN-PRD', 'NP-PN-SBJ-1', 'NP-APP', 'NP-TPC-2', 'NP-PN-SBJ-3', 'NP-PN-IO', 'NP-PN-LOC', 'NP-SBJ-2', 'NP-PN-OBJ-1', 'NP-LGS', 'NP-MNR', 'NP-SBJ-3', 'NP-OBJ-PN', 'NP-SBJ-4', 'NP-PN-SBJ-2', 'NP-TPC-3', 'NP-HLN', 'NP-PN-APP', 'NP-SBJ-PN', 'NP-DIR', 'NP-LOC', 'NP-ADV', 'NP-WH-SBJ']

    if tag in np_list:
        return True
    else:
        return False

def get_info_from_file(file_name,MAX=2):

    pattern = re.compile("(\d+?)\ +(.+?)$")
    pattern_zp = re.compile("(\d+?)\.(\d+?)\-(\d+?)\ +(.+?)$")

    total = 0

    inline = "new"
    f = open(file_name)
    
    sentence_num = 0

    nodes_info = {}   
    candi = {}
    zps = []
    azps = []

    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()

        if line == "Leaves:":
            while True:
                inline = f.readline()
                if inline.strip() == "":break
                inline = inline.strip()
                match = pattern.match(inline)
                if match:
                    word = match.groups()[1]
            sentence_num += 1
    
        elif line == "Tree:":
            candi[sentence_num] = []
            nodes_info[sentence_num] = None
            parse_info = ""
            inline = f.readline()
            while True:
                inline = f.readline()
                if inline.strip("\n") == "":break
                parse_info = parse_info + " " + inline.strip()    
            parse_info = parse_info.strip()            
            nl,wl = parse_analysis.buildTree(parse_info)

            nodes_info[sentence_num] = (nl,wl)

            for node in nl:
                if is_np(node.tag):
                    if node.parent.tag.startswith("NP"):
                        if not (node == node.parent.child[0]):
                            continue
                    leaf_nodes = node.get_leaf()
                    if is_pro(leaf_nodes):
                        continue
                    if is_zero_tag(leaf_nodes):
                        continue
                    candi[sentence_num].append((leaf_nodes[0].index,leaf_nodes[-1].index))
                    total += 1
            for node in wl:
                if node.word == "*pro*":
                    zps.append((sentence_num,node.index))  
 
        elif line.startswith("Coreference chain"):
            first = True
            res_info = None
            last_index = 0
            antecedents = []

            while True:
                inline = f.readline()
                if not inline:break
                if inline.startswith("----------------------------------------------------------------------------------"):
                    break
                inline = inline.strip()
                if len(inline) <= 0:continue
                if inline.startswith("Chain"):
                    first = True
                    res_info = None
                    last_index = 0
                    antecedents = []
                    coref_id = inline.strip().split(" ")[1]
                else:
                    match = pattern_zp.match(inline)
                    if match:
                        sentence_index = int(match.groups()[0])
                        begin_word_index = int(match.groups()[1])
                        end_word_index = int(match.groups()[2])
                        word = match.groups()[-1]
                        if word == "*pro*":
                            is_azp = False
                            if not first:
                                is_azp = True
                                azps.append((sentence_index,begin_word_index,antecedents,coref_id))
                        if not word == "*pro*":
                            first = False
                            res_info = inline
                            last_index = sentence_index
                            antecedents.append((sentence_index,begin_word_index,end_word_index,coref_id))
        
        if not inline:
            break
    return zps,azps,candi,nodes_info
