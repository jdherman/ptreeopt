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
    return '%s < %0.3f' % (self.name, self.threshold)


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


  def validate(self):

    # check if subtree length from the root is equal to the full length
    # (confirms a valid binary tree in this case)
    # this should never happen but is useful for error checking, 
    # sometimes the crossover messes things up

    ix = self.get_subtree(0)
    if len(self.L[ix]) != len(self.L):
      raise RuntimeError('Invalid tree encountered: ' + self)


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

    while node.is_feature:  
      if states[node.index] < node.threshold:
        node = node.l
      else:
        node = node.r

    return node.value


  def get_subtree(self, begin):
    """Adapted from DEAP: return the indices of the subtree
    starting at list index "begin".
    """
    end = begin + 1
    # print self
    # print ' '


    if not self.L[begin].is_feature:
      return slice(begin,end)

    total = 2
    while total > 0:
      if self.L[end].is_feature:
        total += 1
      else:
        total -= 1
      end += 1
      # print 'Begin: %d, End: %d, N: %d, total: %d' % (begin, end, self.N, total)
    return slice(begin, end)


  def get_depth(self):
    # from deap also
    stack = [0]
    max_depth = 0
    for item in self.L:
        depth = stack.pop()
        max_depth = max(max_depth, depth)
        if item.is_feature:
          stack.extend([depth + 1] * 2)
    return max_depth


  # also add "states" to do colors
  def graphviz_export(self, filename = None):

    from graphviz import Digraph
    dot = Digraph(format='png')
    dot.node_attr['shape'] = 'box'
    
    parent = self.root
    dot.node(str(parent), str(parent))
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

      dot.node(str(child), str(child))
      dot.edge(str(parent), str(child), label=label)
      parent = child

    if filename:
      dot.render(filename)
    else:
      import time
      dot.render('graphviz/PTree%s' % time.time())
