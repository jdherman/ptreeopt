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

class PTree:

  def __init__(self, L):
    self.N = len(L)
    self.L = L
    self.validate()
    self.build()


  def validate(self):

    if (self.N & (self.N+1)) != 0:
      raise ValueError('List length + 1 must be a power of 2')


  def build(self):

    self.root = Node(x=self.L[0][0], val=self.L[0][1])
    Q = [self.root]

    for i in range(1, self.N, 2):
      parent = Q.pop(0)

      if parent.is_not_leaf():
        parent.l = Node(*self.L[i])
        parent.r = Node(*self.L[i+1])
        Q += [parent.l, parent.r]


  def evaluate(self, states):

    node = self.root
    
    while node.is_not_leaf():  
      if states[node.x] < node.v:
        node = node.l
      else:
        node = node.r

    return node.v

  # also add "states" to do colors
  def graphviz_export(self, filename = None):

    from graphviz import Digraph

    dot = Digraph(format='png')
    dot.node_attr['shape'] = 'box'
    dot.node(self.root.gvstr(), self.root.gvstr())

    Q = [self.root]

    for i in range(1, self.N, 2):
      parent = Q.pop(0)

      if parent.is_not_leaf():
        parent.l = Node(*self.L[i])
        parent.r = Node(*self.L[i+1])
        Q += [parent.l, parent.r]

        for c in (parent.l, parent.r):
          dot.node(c.gvstr(), c.gvstr())
          dot.edge(parent.gvstr(), c.gvstr())

    if filename:
      dot.render(filename)
    else:
      import time
      dot.render('graphviz/PTree%s.gv' % time.time())





# old code -- not sure if this works anymore
# the reverse of what's happening now:

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
