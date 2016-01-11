import numpy as np

def gen_random_tree_as_list(num_states, max_depth=5):
  depth = np.random.randint(2, max_depth+1)
  num_nodes = 2**depth - 1
  num_leaves = 2**(depth-1)
  L = []

  for i in range(num_nodes):
    if i < num_leaves-1:
      x = np.random.randint(0, num_states)
    else:
      x = None

    L.append((x,np.random.rand()))

  return L