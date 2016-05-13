import numpy as np

# maybe a parameter for tree sparsity (probability of None's)?
def gen_random_tree_as_list(num_states, max_depth=5):
  depth = np.random.randint(2, max_depth+1)
  num_nodes = 2**depth - 1
  num_leaves = 2**(depth-1)
  L = []

  for i in range(num_nodes):
    if i < num_leaves-1:
      x = np.random.randint(0, num_states)
      # what about "val"?
    else:
      x = None

    # if you want an unbalanced tree, some of them will append "none" instead
    L.append((x,np.random.rand()))

  return L

# here: mutation/crossover operators