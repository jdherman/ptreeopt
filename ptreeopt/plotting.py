import numpy as np 
import matplotlib.pyplot as plt
import os, subprocess
import pandas as pd

def graphviz_export(P, filename, colordict=None, animation=False, dpi=300):
    ''' Export policy tree P to filename (SVG or PNG)
    colordict optional. Keys must match actions. Example:
    colordict = {'Release_Demand': 'cornsilk',
            'Hedge_90': 'indianred',
            'Flood_Control': 'lightsteelblue'}
    Requires pygraphviz.'''

    import pygraphviz as pgv
    G = pgv.AGraph(directed=True)
    G.node_attr['shape'] = 'box'
    G.node_attr['style'] = 'filled'

    if animation:
      G.graph_attr['size'] = '2!,2!' # use for animations only
      G.graph_attr['dpi'] = str(dpi)

    parent = P.root
    G.add_node(str(parent), fillcolor='white')
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

        if child.is_feature or not colordict:
          c = 'white'
        else:
          c = colordict[child.value]

        G.add_node(str(child), fillcolor=c)
        G.add_edge(str(parent), str(child), label=label)
        parent = child

    G.layout(prog='dot')
    G.draw(filename)


def animate_trees(snapshots, filename, colordict=None, max_nfe=None):
  
  os.makedirs('temp')

  for i, P in enumerate(snapshots['best_P']):

    nfe = snapshots['nfe'][i]

    if max_nfe and nfe > max_nfe:
        break

    nfestring = 'nfe-' + '%10d' % nfe + '.png'
    graphviz_export(P, 'temp/%s-%s' % (filename, nfestring), colordict, dpi=150)

  subprocess.call(['./ptreeopt/stitch-animations.sh', filename, ''])
  subprocess.call(['rm', '-r', 'temp'])


def ts_color(ts_actions, colordict=None):

  for pol in set(ts_actions):
        first = ts_actions.index[(ts_actions == pol) & (ts_actions.shift(1) != pol)]
        last = ts_actions.index[(ts_actions == pol) & (ts_actions.shift(-1) != pol)]

        for f, l in zip(first, last):
            plt.axvspan(f, l + pd.Timedelta('1 day'),
                        facecolor=colordict[pol], edgecolor='none', alpha=0.4)

def animate_objfxn(snapshots, filename, max_nfe=None):

  os.makedirs('temp')

  for i, P in enumerate(snapshots['best_P']):

    if max_nfe and snapshots['nfe'][i] > max_nfe:
        break

    plt.plot(snapshots['nfe'][:i + 1], snapshots['best_f']
                 [:i + 1], linewidth=2, color='steelblue')

    L = [max_nfe, snapshots['nfe'][-1]]
    plt.xlim([0, min(i for i in L if i is not None)])
    plt.ylim([0, np.max(snapshots['best_f'])])
    plt.ylabel('Objective Function')
    plt.xlabel('NFE')
    plt.tight_layout()

    nfestring = 'nfe-' + '%10d' % snapshots['nfe'][i] + '.png'
    plt.savefig('temp/%s-%s' % (filename, nfestring), dpi=150)
    plt.close()

  subprocess.call(['./ptreeopt/stitch-animations.sh', filename, '-layers optimize'])
  subprocess.call(['rm', '-r', 'temp'])

