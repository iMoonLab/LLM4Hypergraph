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

"""Library for encoding graphs in text."""

import name_dictionaries
import networkx as nx 
NODE_ENCODER_DICT = {
    "N-Pair":{k:'v'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "LO-Inc":{k:'v'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "Adj-Mat":{k:'v'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "N-Set": {k:'v'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "HO-Inc":{k:'v'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "Inc-Mat": {k:'v'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "HO-Neigh":{k:'v'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
}
EDGES_ENCODER_DICT = {
    "N-Pair":{k:'e'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "LO-Inc":{k:'e'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "Adj-Mat":{k:'e'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "N-Set": {k:'e'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "HO-Inc":{k:'e'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "Inc-Mat": {k:'e'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
    "HO-Neigh":{k:'e'+v for k,v in name_dictionaries.create_name_dict("integer").items()},
}

def create_vertex_string(name_dict, nvertices):
  vertex_string = ""
  for i in range(nvertices - 1):
    vertex_string += name_dict[i] + ", "
  vertex_string += "and " + name_dict[nvertices - 1]
  return vertex_string

def create_hyperedge_string(nvertices,edge_dict):
  vertex_string = ""
  for i in range(nvertices - 1):
    vertex_string += str(edge_dict[i]) + ", "
  vertex_string += "and " + str(edge_dict[nvertices - 1])
  return vertex_string



def N_Set_encoder(graph, name_dict,edge_dcit):
  output = (
      "In an undirected hypergraph, (i, j, k) means that vertex i, vertex j and vertex k are"
      " connected with an undirected hyperedge. "
  )
  vertices_string = create_vertex_string(name_dict, len(graph.v))
  edges_string = create_vertex_string(edge_dcit,len(graph.e[0]))
  output += "G describes a hypergraph among vertices %s " % vertices_string
  output += "and among hyperedges %s.\n"% edges_string
  if graph.e[0]:
    output += "The hyperedges in G are: "
  for edge in graph.e[0]:
    tmp = ''
    for i in edge:
      tmp += "%s," % (name_dict[i])
    tmp = '(' + tmp[:-1] + '),'
    output += tmp
  return output.strip() + ".\n"



def HO_Inc_encoder(graph, name_dict,edge_dcit):
  "Encoding a hypergraph with its clique expanation graph with incident list."
  num_v, edges = graph.clique_expanation()
  vertices_string = create_vertex_string(name_dict, num_v)
  edges_string = create_hyperedge_string(nvertices=len(graph.e[0]),edge_dict=edge_dcit)
  output = "G describes a hypergraph among vertice %s and among hyperedges %s.\n" % (vertices_string ,edges_string )
  if edges:
    output += "In this hypergraph:\n"
  
  for source_vertex in range(num_v):
    neibor_edges = graph.edges(source_vertex)
    output += "vertex %s is connected" % name_dict[source_vertex]
    for e in neibor_edges:
      target_vertices = []
      edge = graph.e[0][e]
      for vertex in edge:
        if vertex != source_vertex:
          target_vertices.append(vertex)
      target_vertices_str = ""
      nedges = 0
      for target_vertex in target_vertices:
        target_vertices_str += name_dict[target_vertex] + ", "
        nedges += 1
      if nedges > 1:
        output += " to vertice %s with hyperedge %s," % (
            target_vertices_str[:-2],
            edge_dcit[e]
        )
      elif nedges == 1:
        output += " to vertex %s with hyperedge %s," % (
            target_vertices_str[:-2],
            edge_dcit[e]
        )
    output = output[:-1] + ".\n"
  return output


def HO_Neigh_encoder(graph, name_dict,edge_dcit):
  vertices_string = create_vertex_string(name_dict, len(graph.v))
  edge_string = create_vertex_string(edge_dcit,len(graph.e[0]))
  output = f"G describes a hypergraph among vertices {vertices_string} and hyperedges {edge_string}.\n"
  if graph.e[0]:
    output += "In this hypergraph:\n"

  for source_vertex in graph.v:
    tmp = []
    for j,edge in enumerate(graph.e[0]):
      if source_vertex in edge:
        tmp.append(j)
    if len(tmp) > 1: 
      output += f"vertex {name_dict[source_vertex]} is connected to hyperedges "
      for i in tmp:
        output += f'{edge_dcit[i]},'
      output = output[:-1] + '.\n'
    elif len(tmp) == 1: 
      output += f"vertex {name_dict[source_vertex]} is connected to hyperedges {edge_dcit[tmp[0]]}.\n"
    else: 
      pass       
  for k,source_edge in enumerate(graph.e[0]):
    output += f'Hyperedge {edge_dcit[k]} is connected to vertices '
    for n in source_edge:
      output += f'{name_dict[n]},'
    output = output[:-1] + '.\n'
  return output



def N_Pair_encoder(graph, name_dict,edge_dcit):
  """Encoding a hypergraph with its clique expanation graph with Adjacency"""
  num_v, edges = graph.clique_expanation_low()
  vertices_string = create_vertex_string(name_dict, num_v)
  edge_string = create_vertex_string(edge_dcit,len(graph.e[0]))
  output = (
        "In an undirected hypergraph, (i,j) means that vertex i and vertex j are"
        " connected with an undirected hyperedge. "
    )
  output += f"G describes a hypergraph among vertices {vertices_string} and hyperedges {edge_string}.\n"
  if edges:
    output += "The connection relation between vertices in G are: "
  for i, j in edges:
    output += "(%s, %s) " % (name_dict[i], name_dict[j])
  return output.strip() + ".\n"


def LO_Inc_encoder(graph, name_dict,edge_dcit):
  "Encoding a hypergraph with its clique expanation graph with incident list."
  num_v, edges = graph.clique_expanation_low()
  vertices_string = create_vertex_string(name_dict, num_v)
  edge_string = create_vertex_string(edge_dcit,len(graph.e[0]))
  output = f"G describes a hypergraph among vertices {vertices_string} and hyperedges {edge_string}.\n"
  if edges:
    output += "In this hypergraph:\n"
  
  for source_vertex in range(num_v):
    target_vertices = graph.clique_neighbor_low(source_vertex)
    target_vertices_str = ""
    nedges = 0
    for target_vertex in target_vertices:
      target_vertices_str += name_dict[target_vertex] + ", "
      nedges += 1
    if nedges > 1:
      output += "vertex %s is connected to vertices %s.\n" % (
          name_dict[source_vertex],
          target_vertices_str[:-2],
      )
    elif nedges == 1:
      output += "vertex %s is connected to vertex %s.\n" % (
          name_dict[source_vertex],
          target_vertices_str[:-2],
      )
  return output


def Inc_Mat_encoder(graph,name_dict,edge_dcit):
  num_v, edges = len(graph.v) , graph.e[0]
  vertices_string = create_vertex_string(name_dict, len(graph.v))
  edge_string = create_vertex_string(edge_dcit,len(graph.e[0]))
  output = f"G describes a hypergraph among vertices {vertices_string} and hyperedges {edge_string}.\n"
  if edges:
    output += "The incidence matrix of the hypergraph is\n"
  def get_adj_matrix(hypergraph):
    H = hypergraph.H.to_dense().int().numpy()
    H_matrix_str = "["
    for i in range(H.shape[0]):
      tmp = "["
      for j in range(H.shape[1]):
        tmp += str(H[i,j])
        tmp += ","
      tmp += '],\n'
      H_matrix_str += tmp 
    H_matrix_str = H_matrix_str[:-2] + "]\n"
    return H_matrix_str
  output += get_adj_matrix(graph)
  return output

def Adj_Mat_encoder(graph,name_dict,edge_dcit):
  num_v, edges = len(graph.v) , graph.e[0]
  vertices_string = create_vertex_string(name_dict, len(graph.v))
  edge_string = create_vertex_string(edge_dcit,len(graph.e[0]))
  output = f"G describes a hypergraph among vertices {vertices_string} and among hyperedges {edge_string}.\n"
  if edges:
    output += "The adjacency matrix between vertices of the hypergraph is\n"
  def get_clique_adj_matrix(hypergraph):
    H = hypergraph.H.to_dense().int()
    H = H @ H.T
    H = H.bool().int().numpy()
    H_matrix_str = "["
    for i in range(H.shape[0]):
      tmp = "["
      for j in range(H.shape[1]):
        tmp += str(H[i,j])
        tmp += ","
      tmp += '],\n'
      H_matrix_str += tmp 
    H_matrix_str = H_matrix_str[:-2] + "]\n"
    return H_matrix_str
  output += get_clique_adj_matrix(graph)
  return output
  

TEXT_ENCODER_FN = {
  "N-Pair":N_Pair_encoder,
  "LO-Inc":LO_Inc_encoder,
  "Adj-Mat":Adj_Mat_encoder,
  "N-Set": N_Set_encoder,
  "HO-Inc":HO_Inc_encoder,
  "Inc-Mat": Inc_Mat_encoder,
  "HO-Neigh":HO_Neigh_encoder,
}


def with_ids(graph, text_encoder):
  nx.set_node_attributes(graph, NODE_ENCODER_DICT[text_encoder], name="id")
  return graph


def encode_graph(graph, text_encoder):
  """Encoding a graph according to the given text_encoder method."""
  name_dict = NODE_ENCODER_DICT[text_encoder]
  edge_dict = EDGES_ENCODER_DICT[text_encoder]
  return TEXT_ENCODER_FN[text_encoder](graph, name_dict,edge_dict)
