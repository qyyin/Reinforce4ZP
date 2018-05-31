#coding=utf8
import sys
import os
import re

class Node:
    tag = None
    parent = None
    index = -1
    word = ""
    child = []
    right = None
    left = None
    def __init__(self,parent=None,word="",tag="",index=-1):
        self.parent = parent
        self.word = word
        self.tag = tag
        self.child = []
        self.index = index
        self.right = None
        self.left = None
    def has_child(self,n):
        if n in self.child:
            return True
        return False
    def add_child(self,child):
        self.child.append(child)
    def get_leaf(self):
        nl = []
        for c in self.child:
            if c.index >= 0:
                nl.append(c)
            else:
                nl += c.get_leaf()
        return nl
    def get_pub_node(self,node):
        if not node:
            return None
        father = node.parent
        while True:
            if not father:break
            sf = self.parent
            while True:
                if not sf:break
                if sf == father:
                    return father
                sf = sf.parent
            father = father.parent
        return None

class Stack:
    items = []
    def __init__(self,items=[]):
        self.items = items
    def push(self,item):
        self.items.append(item)
    def size(self):
        return len(self.items)
    def pop(self):
        if len(self.items) > 0:
            last = self.items[-1]
            self.items = self.items[:-1]
            return last
        else:
            return None
    def last(self):
        if len(self.items) > 0:
            return self.items[-1]
        else:
            return None
    def combine(self):
        return self.items

def print_node_list(nl):
    for n in nl:
        if n.parent:
            print "******"
            print n.index,n.word,n.tag,n.parent.tag
            print "child:"
            for q in n.child:
                print q.tag,
            print
        else:
            print "******"
            print n.index,n.word,n.tag,"None"
            print "child:"
            for q in n.child:
                print q.tag,
            print

def buildTree(parse):
    stack = Stack([])
    item = ""
    parent = None
    left = None
    right = None
    nl = []
    wl = []
    word_index = 0
    for letter in parse.decode("utf8"):
        if letter == "(":
            if len(item.strip()) > 0:
                item = item.strip().encode("utf8")
                item = item.split(" ")
                word = ""
                tag = ""
                index = -1
                if len(item) == 2:
                    #word
                    tag = item[0].strip()
                    word = item[1].strip()
                    index = word_index
                    word_index += 1
                else:
                    tag = item[0].strip()
                node = Node(parent,word,tag,index)

                if len(item) == 2:
                    node.left = left
                    wl.append(node)
                    if node.left:
                        node.left.right = node
                    left = node

                if node.parent:
                    node.parent.add_child(node)
                stack.push(parent)
                parent = node
                nl.append(node)
                item = ""
        elif letter == ")":
            if len(item.strip()) > 0:
                item = item.strip().encode("utf8")
                item = item.split(" ")
                word = ""
                tag = ""
                index = -1
                if len(item) == 2:
                    #word
                    tag = item[0].strip()
                    word = item[1].strip()
                    index = word_index
                    word_index += 1
                else:
                    tag = item[0].strip()
                node = Node(parent,word,tag,index)

                if len(item) == 2:
                    node.left = left
                    wl.append(node)
                    if node.left:
                        node.left.right = node
                    left = node

                if node.parent:
                    node.parent.add_child(node)
                nl.append(node)
                item = ""
            else:
                last = stack.pop()
                parent = last
        else:
            item += letter
    return nl,wl 
