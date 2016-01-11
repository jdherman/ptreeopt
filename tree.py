import numpy as np

class Node:
  def __init__(self, x, val):
    self.l = None
    self.r = None
    self.x = x
    self.v = val

  def __str__(self):
      return str(self.v)

def build_tree(L):
  # assumes a full binary tree, given in level order
  # do some error checking for length here, then
  root = Node(x=L[0][0], val=L[0][1])
  Q = [root]
  i = 1
  while i < len(L):
    parent = Q.pop(0)
    left_child = Node(x=L[i][0], val=L[i][1])
    Q.append(left_child)
    i += 1
    if i < len(L):
      right_child = Node(x=L[i][0], val=L[i][1])
      Q.append(right_child)
      i += 1
    else:
      right_child = None
    parent.l = left_child
    parent.r = right_child
  return root

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

# def export_graphviz

# def gini (or some other importance measure)


def eval_tree(node, states):
  while node.x:
    if states[node.x] < node.v:
      node = node.l
    else:
      node = node.r
  return node.v


# L = [(1,10), (5,12), (3,15), (2,25), (4,30), (0,36)]
L = [(1,10), (None,25), (None,36)]
T = build_tree(L) # do this once at the beginning of the simulation
# L = gen_random_tree_as_list(3, 5)
print eval_tree(T, states=[2,11,100]) # do this every timestep
