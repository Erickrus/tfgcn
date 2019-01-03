import networkx as nx
import numpy as np

class GCNUtil:

  # get the adjacency matrix
  def adjacency(self, graph, size=16):
    res = []
    nodes = sorted(list(graph.nodes))
    for i in range(size):
      row = []  
      for j in range(size):
        row.append(0.)
      res.append(row)

    for j in range(len(nodes)):
      node = nodes[j]
      neighbors = list(graph.neighbors(node))
      for i in range(len(neighbors)):
        p = neighbors[i]
        pos = nodes.index(p)
        res[j][pos] = 1.0
    return np.array(res)

  # get degree matrix
  def degree(self, graph, size=16):
    res = []

    nodes = sorted(list(graph.nodes))
    degLen = len(nodes)
    for i in range(size):
      if i+1 <= degLen:
        node = nodes[i]
        res.append(len(list(graph.neighbors(node))))
      else:
        res.append(0.0)
      
    return np.diag(np.array(res))

  # get feature from matrix
  def feature(self, adj):
    _, x = np.linalg.eig(adj)
    return x
    #x = np.ones(adj.shape)
    #return x

  def get_subgraph(self, graph, node, hop):
    subgraph = nx.Graph()
    visitedNodes = {}
    exploringNodes = [node]
    for i in range(hop+1):
      nextExploringNodes = []
      for currNode in exploringNodes:
        visitedNodes[currNode] = 1
        for neighborNode in graph.neighbors(currNode):
          if (not neighborNode in visitedNodes and 
              not neighborNode in exploringNodes and 
              not neighborNode in nextExploringNodes):
            nextExploringNodes.append(neighborNode)
      exploringNodes = nextExploringNodes 
    nodes = list(visitedNodes.keys())

    for node in nodes:
      subgraph.add_node(node)

    for edge in graph.edges:
      if edge[0] in nodes and edge[1] in nodes:
        subgraph.add_edge(edge[0], edge[1])

    return subgraph


  

