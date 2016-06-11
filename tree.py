class Node(object):

  def __init__(self):
    self.l = None
    self.r = None

  def __str__(self):
    return str(self.v)


class Feature(Node):

  def __init__(self, contents):
    self.index, self.threshold = contents
    self.is_feature = True
    super(Feature, self).__init__()

  def __str__(self):
    return 'X[%d] < %s' % (self.index, self.threshold)


class Action(Node):

  def __init__(self, contents):
    self.value = contents[0]
    self.is_feature = False
    super(Action, self).__init__()

  def __str__(self):
    return '%s' % self.value


class PTree:

  def __init__(self, L):
    self.N = len(L)
    self.L = [Feature(item) if len(item)==2 else Action(item) for item in L]
    self.root = None
    # self.validate()
    self.build()


  # def validate(self):

  #   if (self.N & (self.N+1)) != 0:
  #     raise ValueError('List length + 1 must be a power of 2')


  def build(self):
    self.root = self.L[0]
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

      else:
        parent = S.pop()
        child = parent.r

      dot.node(str(child), str(child))
      dot.edge(str(parent), str(child))
      parent = child

    if filename:
      dot.render(filename)
    else:
      import time
      dot.render('graphviz/PTree%s.gv' % time.time())
