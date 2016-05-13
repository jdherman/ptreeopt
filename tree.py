import numpy as np
from graphviz import Digraph
import time

class Node:
  def __init__(self, x, val):
    self.l = None
    self.r = None
    self.x = x
    self.v = val

  def __str__(self):
    return str(self.v)

  def is_not_leaf(self):
    return self.x is not None

  def gvstr(self):
    if self.is_not_leaf():
      return 'X[%d] <= %s' % (self.x,self.v)
    else:
      return '%s' % self.v


def validate_tree(L):
  N = len(L)
  if (N & (N+1)) != 0:
    raise ValueError('List length + 1 must be a power of 2')

  # some other really important stuff here ...


def build_tree(L, export_png=False):
  # assumes list has length of a full binary tree, given in level order
  # but there can be "none" nodes which are not added

  N = len(L)
  validate_tree(L)

  root = Node(x=L[0][0], val=L[0][1])
  if export_png:
    dot = Digraph(format='png')
    dot.node_attr['shape'] = 'box'
    dot.node(root.gvstr(), root.gvstr())

  Q = [root]
  i = 1
  while i < N:
    parent = Q.pop(0)
    lchild,rchild = (None,None)

    if L[i]:
      lchild = Node(x=L[i][0], val=L[i][1])
      Q.append(lchild)

    if L[i+1]:
      rchild = Node(x=L[i+1][0], val=L[i+1][1])
      Q.append(rchild)
    
    i += 2

    if lchild and rchild:
      parent.l = lchild
      parent.r = rchild

      if export_png:
        for c in (lchild,rchild):
          dot.node(c.gvstr(), c.gvstr())
          dot.edge(parent.gvstr(), c.gvstr())

  if export_png:
    dot.render('graphviz/Tree%s.gv' % time.time())

  return root


def eval_tree(node, states):
  while node.is_not_leaf():
    if states[node.x] < node.v:
      node = node.l
    else:
      node = node.r
  return node.v







# old code -- not sure if this works anymore

# def tree_to_list(root):
#   # level order traversal
#   Q = [root]
#   L = []
#   while Q:
#     parent = Q.pop(0)
#     L.append((parent.x, parent.v))
#     if parent.l:
#       Q.append(parent.l)
#     if parent.r:
#       Q.append(parent.r)
#   return L

# def __str__(self):
#   return str(self.L)
