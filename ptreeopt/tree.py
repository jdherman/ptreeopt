class Node(object):
    '''
    
    Attributes
    ----------
    l : 
    r : 
    
    '''

    def __init__(self):
        self.l = None
        self.r = None

    def __str__(self):
        raise NotImplementedError('Must be defined in a child class')


class Feature(Node):
    '''
    
    Attributes
    ----------
    index : 
    threshold :
    name : str
    is_feature : bool
    is_discrete : bool
    
    
    '''

    def __init__(self, contents):
        self.index, self.threshold = contents
        self.name = 'X[%d]' % self.index
        self.is_feature = True
        self.is_discrete = False
        super(Feature, self).__init__()

    def __str__(self):
        if self.is_discrete:
            return '%s == %d' % (self.name, self.threshold)
        else:
            return '%s < %d' % (self.name, self.threshold)


class Action(Node):

    def __init__(self, contents):
        self.value = contents[0]
        self.is_feature = False
        self.count = -1
        super(Action, self).__init__()

    def __str__(self):
        try:
            return '%0.3f (%0.2f%%)' % (self.value, self.count)
        except TypeError:
            return '%s (%0.2f%%)' % (self.value, self.count)


class PTree(object):
    '''
    
    Attributes
    ----------
    L : list of Feature instances
    root : 
    
    
    
    '''

    def __init__(self, L, feature_names=None, discrete_features=None):
        self.L = []

        for item in L:
            if len(item) == 2:
                f = Feature(item)
                if feature_names:
                    f.name = feature_names[f.index]
                if discrete_features:
                    f.is_discrete = discrete_features[f.index]
                    f.threshold = int(round(f.threshold)) # round
                self.L.append(f)
            else:
                self.L.append(Action(item))

        self.root = None
        self.build()

    def __str__(self):
        return ', '.join([str(item) for item in self.L])

    def __getitem__(self, i):
        return self.L[i]

    def __setitem__(self, i, v):
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
            if (node.is_discrete and states[node.index] == node.threshold) \
                or (not node.is_discrete and states[node.index] < node.threshold):
                rules.append((node.name, node.threshold, True))
                node = node.l
            else:
                rules.append((node.name, node.threshold, False))
                node = node.r

        node.count += 1 # track number of action occurrences
        return (node.value, rules)

    def clear_count(self):
        # reset action counts to zero
        for node in self.L:
            if not node.is_feature:
                node.count = 0

    def normalize_count(self):
        # convert action counts to percents
        s = sum([node.count for node in self.L if not node.is_feature])
        for node in self.L:
            if not node.is_feature:
                node.count /= s/100

    def get_subtree(self, begin):
        # Adapted from DEAP: return the indices of the subtree
        # starting at list index `begin`.
        end = begin + 1

        if not self.L[begin].is_feature:
            return slice(begin, end)

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

            l = self.get_subtree(i + 1)
            r = self.get_subtree(l.stop)

            if self._prune_subtree(i, r, mode='right') or \
               self._prune_subtree(i, l, mode='left') or \
               self._prune_actions(i, l, r):
                continue

            i += 1

        self.build()

    def _prune_subtree(self, i, s, mode):
        # Removes illogical subtree relationships by hoisting subtrees

        current = self[i]

        def _hoist_subtree(j, side):
            if side == 'r':
                sub = self.get_subtree(self.get_subtree(j + 1).stop)
            else:
                sub = self.get_subtree(j + 1)
            self[self.get_subtree(j)] = self[sub]

        # loop over nodes in a subtree, represented by a list slice
        for j in range(s.start, s.stop):

            child = self[j]

            if child.is_feature and child.index == current.index:

                if child.is_discrete: # discrete features
                    '''If a feature in the right subtree has equal threshold, 
                    it's false. If feature in left subtree has a not-equal 
                    threshold, it's false. Both cases: replace with right subtree'''
                    if (mode == 'right' and child.threshold == current.threshold) \
                    or (mode == 'left' and child.threshold != current.threshold):
                        _hoist_subtree(j, 'r')
                        return True

                    '''If a feature in the left subtree has equal threshold,
                    we already know it's true (replace with left subtree)'''
                    if (mode == 'left' and child.threshold == current.threshold):
                        _hoist_subtree(j, 'l')
                        return True

                else: # continuous features
                    '''If a feature in the right subtree has a threshold 
                    less than current,replace it with its own right subtree'''
                    if (mode == 'right' and child.threshold < current.threshold):
                        _hoist_subtree(j, 'r')
                        return True
                    '''If a feature in the left subtree has a threshold greater
                     than current, replace it with its left subtree'''
                    if (mode == 'left' and child.threshold > current.threshold):
                        _hoist_subtree(j, 'l')
                        return True

        return False

    def _prune_actions(self, i, l, r):
        # two cases: prune duplicate actions, and unused actions
        lchild = self[l][0]
        rchild = self[r][0]
        pruned = False

        if not lchild.is_feature and \
            not rchild.is_feature and i != 0:

            if lchild.value == rchild.value or rchild.count == 0:
                self.L[i] = lchild
                pruned = True
            elif lchild.count == 0:
                self.L[i] = rchild
                pruned = True

        if pruned:
            self.L[r] = []  # MUST delete right one first
            self.L[l] = []
            return True

        return False

