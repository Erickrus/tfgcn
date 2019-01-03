import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class ToyDataset:
  def get_naive_graph(self):
    g = nx.Graph()
    for i in range(4):
      g.add_node(i)
    for edge in '0-0,0-1,0-3,1-3,1-2,2-3'.split(','):
      fromNode = int(edge.split('-')[0])
      toNode   = int(edge.split('-')[1])
      g.add_edge(fromNode, toNode)
    return g

  def get_graph(self):
    numNode = 16
    edges = '1-2,1-4,1-3,2-2,2-4,3-4,'+ \
            '4-5,5-6,5-7,5-8,7-7,7-8,'+ \
            '6-8,6-9,9-12,9-11,9-10,'+ \
            '11-12,11-10,8-13,13-14,'+ \
            '13-16,14-16,14-15,15-16,15-15'
    g = nx.Graph()
    for i in range(numNode):
      g.add_node(i+1)
    for edge in edges.split(','):
      fromNode = int(edge.split('-')[0])
      toNode   = int(edge.split('-')[1])
      g.add_edge(fromNode, toNode)
    return g

  def get_label(self):
    res = 'bbbccefecaaacddd'
    labels = []
    for ch in res:
      labels.append(ch)
    # https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    labels = LabelEncoder().fit_transform(labels)
    labels = labels.reshape(len(labels), 1)
    res = OneHotEncoder(sparse=False).fit_transform(labels)
    return res

  # https://stackoverflow.com/questions/39657395/how-to-draw-properly-networkx-graphs
  # https://stackoverflow.com/questions/40528048/pip-install-pygraphviz-no-package-libcgraph-found
  def draw(self, graph, filename):
    g = to_agraph(graph)
    g.layout('dot')
    g.draw(filename)

if __name__ == "__main__":
  tds = ToyDataset()
  g = tds.get_graph()
  tds.get_label()

