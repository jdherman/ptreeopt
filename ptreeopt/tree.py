class Node(object):

  def __init__(self):
    self.l = None
    self.r = None

  def __str__(self):
    raise NotImplementedError('Must be defined in a child class')


class Feature(Node):

  def __init__(self, contents):
    self.index, self.threshold = contents
    self.name = 'X[%d]' % self.index
    self.is_feature = True
    super(Feature, self).__init__()

  def __str__(self):
    return '%s < %d' % (self.name, self.threshold)


class Action(Node):

  def __init__(self, contents):
    self.value = contents[0]
    self.is_feature = False
    super(Action, self).__init__()

  def __str__(self):
    try:
      return '%0.3f' % self.value
    except TypeError:
      return self.value


class PTree:

  def __init__(self, L, feature_names=None):
    self.L = []
    
    for item in L:
      if len(item)==2:
        f = Feature(item)
        if feature_names:
          f.name = feature_names[f.index]
        self.L.append(f)  
      else:
        self.L.append(Action(item))

    self.root = None
    self.build()


  def __str__(self):
    return ', '.join([str(item) for item in self.L])

  def __getitem__(self,i):
    return self.L[i]

  def __setitem__(self,i,v):
    self.L[i] = v


  def validate(self):

    # check if subtree length from the root is equal to the full length
    # (confirms a valid binary tree in this case)
    # this should never happen but is useful for error checking, 
    # sometimes the crossover messes things up

    ix = self.get_subtree(0)
    if len(self.L[ix]) != len(self.L):
      raise RuntimeError('Invalid tree encountered: ' + str(self))


  def build(self):

    self.root = self.L[0]
    self.N = len(self.L)
    self.validate()
    parent = self.root

    S = []
    
    for child in self.L:

      if parent.is_feature:
        parent.l = child
        S.append(parent)

      elif len(S) > 0:
        parent = S.pop()
        parent.r = child

      parent = child


  def evaluate(self, states):

    node = self.root
    rules = []

    while node.is_feature:
      if states[node.index] < node.threshold:
        rules.append((node.name, node.threshold, True))  
        node = node.l
      else:
        rules.append((node.name, node.threshold, False))  
        node = node.r

    return (node.value, rules)


  def get_subtree(self, begin):
    # Adapted from DEAP: return the indices of the subtree
    # starting at list index `begin`.
    end = begin + 1

    if not self.L[begin].is_feature:
      return slice(begin,end)

    total = 2
    while total > 0:
      if self.L[end].is_feature:
        total += 1
      else:
        total -= 1
      end += 1
    return slice(begin, end)


  def get_depth(self):
    # Adapted from DEAP
    stack = [0]
    max_depth = 0
    for item in self.L:
        depth = stack.pop()
        max_depth = max(max_depth, depth)
        if item.is_feature:
          stack.extend([depth + 1] * 2)
    return max_depth


  def prune(self):

    i = 0

    while i < len(self.L):

      if not self[i].is_feature:
        i += 1
        continue

      l = self.get_subtree(i+1)
      r = self.get_subtree(l.stop)

      if self._prune_subtree(i, r, mode='right') or \
         self._prune_subtree(i, l, mode='left') or \
         self._prune_duplicate_actions(i, l, r):
        continue

      i += 1

    self.build()


  def _prune_subtree(self, i, s, mode):
    '''Removes illogical subtree relationships.
    If a feature in the right subtree has a threshold less than current,
    Replace it with its own right subtree. If a feature in the left
    subtree has a threshold greater than current, replace it with its left subtree.'''

    current = self[i]

    for j in range(s.start, s.stop):
      
      child = self[j]

      if child.is_feature and child.index == current.index:
        
        if mode == 'right' and child.threshold < current.threshold:
          rsub = self.get_subtree(self.get_subtree(j+1).stop)
          self[self.get_subtree(j)] = self[rsub]
          return True

        elif mode == 'left' and child.threshold > current.threshold:
          lsub = self.get_subtree(j+1)
          self[self.get_subtree(j)] = self[lsub]
          return True

    return False


  def _prune_duplicate_actions(self, i, l, r):

    lchild = self[l][0]
    rchild = self[r][0]

    if not lchild.is_feature and \
       not rchild.is_feature and \
       i != 0 and \
       lchild.value == rchild.value:
        self.L[i] = lchild
        self.L[r] = [] # MUST delete right one first
        self.L[l] = []
        return True

    return False


  # also add "states" to do colors
  def graphviz_export(self, filename, dpi = 300):

    import pygraphviz as pgv
    G = pgv.AGraph(directed=True)
    G.node_attr['shape'] = 'box'
    # G.graph_attr['size'] = '2!,2!' # use for animations only
    # G.graph_attr['dpi'] = str(dpi)
    
    parent = self.root
    G.add_node(str(parent))
    S = []

    while parent.is_feature or len(S) > 0:
      if parent.is_feature:
        S.append(parent)
        child = parent.l
        label = 'T'

      else:
        parent = S.pop()
        child = parent.r
        label = 'F'

      G.add_node(str(child))
      G.add_edge(str(parent), str(child), label=label)
      parent = child

    G.layout(prog='dot')
    G.draw(filename)
