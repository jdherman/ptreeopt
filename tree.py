class Node(object):

  def __init__(self):
    self.l = None
    self.r = None

  def __str__(self):
    return str(self.v)


class Feature(Node):

  def __init__(self, contents):
    self.index, self.threshold = contents
    self.arity = 2
    super(Feature, self).__init__()

  def __str__(self):
    return 'X[%d] < %s' % (self.index, self.threshold)


class Action(Node):

  def __init__(self, contents):
    self.value = contents[0]
    self.arity = 0
    super(Action, self).__init__()

  def __str__(self):
    return '%s' % self.value


class PTree:

  def __init__(self, L):
    self.N = len(L)
    self.L = L
    # self.validate()
    self.build()


  # def validate(self):

  #   if (self.N & (self.N+1)) != 0:
  #     raise ValueError('List length + 1 must be a power of 2')


  def build(self):

    self.root = Feature(self.L[0])
    S = [self.root]
    parent = self.root

    i = 1
    
    while i < self.N:

      if parent.arity == 2:

        child = self.L[i]
        if parent.l is None:

          if len(child) == 2:
            parent.l = Feature(child)
            S.append(parent.l)
          else:
            parent.l = Action(child)

          parent = parent.l

        else:
          if len(child) == 2:
            parent.r = Feature(child)
            S.append(parent.r)
          else:
            parent.r = Action(child)

          parent = parent.r

        i += 1

      else:

        if len(S) > 0:
          parent = S.pop()

      


  def evaluate(self, states):

    node = self.root
    
    while node.arity == 2:  
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

    while parent.arity == 2 or len(S) > 0:

      if parent.arity == 2:
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
