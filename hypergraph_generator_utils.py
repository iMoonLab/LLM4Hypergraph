# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Random graph generation."""

import random

import networkx as nx
import numpy as np
import dhg 



_NUMBER_OF_NODES_RANGE = {
    "small": np.arange(5, 10),
    "medium": np.arange(10, 15),
    "large": np.arange(15, 20),
}
_NUMBER_OF_COMMUNITIES_RANGE = {
    "small": np.arange(2, 4),
    "medium": np.arange(2, 8),
    "large": np.arange(2, 10),
}


def generate_graphs(
    number_of_graphs,
    algorithm,
    directed,
    random_seed = 1234,
    er_min_sparsity = 0.0,
    er_max_sparsity = 1.0,
):
  """Generating multiple graphs using the provided algorithms.

  Args:
    number_of_graphs: number of graphs to generate
    algorithm: the random graph generator algorithm
    directed: whether to generate directed or undirected graphs.
    random_seed: the random seed to generate graphs with.
    er_min_sparsity: minimum sparsity of er graphs.
    er_max_sparsity: maximum sparsity of er graphs.

  Returns:
    generated_graphs: a list of nx graphs.
  Raises:
    NotImplementedError: if the algorithm is not yet implemented.
  """

  random.seed(random_seed)
  np.random.seed(random_seed)

  generated_graphs = []
  graph_sizes = random.choices(
      list(_NUMBER_OF_NODES_RANGE.keys()), k=number_of_graphs
  )
  random_state = np.random.RandomState(random_seed)
  if algorithm == 'hypergraph':
    for i in range(number_of_graphs):
      number_of_vertices = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes[i]])
      number_of_hypedges = random.choice(range(int(number_of_vertices*0.2),int(number_of_vertices*1.5)))
      sparsity = [random.uniform(er_min_sparsity, er_max_sparsity) for i in range(number_of_hypedges)]
      generated_graphs.append(
        dhg.random.hypergraph_Gnm(num_v=int(number_of_vertices),num_e=number_of_hypedges)
      )
  elif algorithm == "graph1":
    for i in range(number_of_graphs):
      number_of_vertices = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes[i]])
      number_of_hypedges = random.choice(range(int(number_of_vertices*0.2),int(number_of_vertices*1.5))) # TODO 记得改点的数目和边的数目
      prob_k_list = [0 for k in range(number_of_vertices-1)]
      prob_k_list[0] = 1 #
      generated_graphs.append(
        dhg.random.hypergraph_Gnm(num_v=int(number_of_vertices),num_e=number_of_hypedges,method="custom",prob_k_list=prob_k_list)
      )
  elif algorithm == "graph2":
    for i in range(number_of_graphs):
      number_of_vertices = random.choice(np.arange(5, 10))
      # number_of_vertices = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes[i]])
      g = dhg.random.graph_Gnp(num_v=int(number_of_vertices),prob=random.random())
      while  len(g.e[0]) == 0:
        g = dhg.random.graph_Gnp(num_v=int(number_of_vertices),prob=random.random())
      g = dhg.structure.Hypergraph.from_graph(g)
      generated_graphs.append(
        g
      )
  elif algorithm == 'hypergraph_high':
    for i in range(number_of_graphs):
      number_of_vertices = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes[i]])
      number_of_hypedges = random.choice(range(int(number_of_vertices*0.2),int(number_of_vertices*1.5)))
      prob_k_list = [2 ** (-k) for k in range(number_of_vertices-1)]
      generated_graphs.append(
        dhg.random.hypergraph_Gnm(num_v=int(number_of_vertices),num_e=number_of_hypedges,method="custom",prob_k_list=prob_k_list)
      )
  else:
    raise NotImplementedError()
  return generated_graphs




def random_hypergraph(n,e,p,seed):
  """
  n : number of vertices in hypergraph 
  e: number of hypedges in hypergraph 
  p: the probability of a vertex belong to a edges 
  """
  G = {}
  G['vertex'] = list(range(n))
  G['hypedges'] = []
  for i in range(e):
    edge = [] 
    for vertex in range(n):
      if seed.random() < p[i]:
        edge.append(vertex)
    if len(edge) > 0 and edge not in G['hypedges']:
      G['hypedges'].append(edge)
  return G

import pickle
def write_graph_pkl(HypeGraph,path):
  with open(path, 'wb') as f:
    pickle.dump(HypeGraph, f)
  
def load_graph_pkl(HypeGraph,path):
  with open(path, 'rb') as f:
    loaded_data = pickle.load(f)
  return loaded_data


