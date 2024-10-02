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

"""The graph tasks to be tried with LLMs."""

import random

import networkx as nx
import numpy as np

# from graphqa import graph_text_encoder
from hyper_graph import HyperGraph
import hypergraph_text_encoder
class GraphTask:
  """The parent class for all the graph tasks."""

  def __init__(self):
    self.name = 'default'
    self.maximum_nvertices_cot_graph = 10

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    raise NotImplementedError()

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    raise NotImplementedError()



class VertexConnectionCheck(GraphTask):
  """The hypergraph task to check if vertex a connected to vertex b"""

  def __init__(self):
    super().__init__()
    self.name = 'vertex_connection'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.v), k=2)
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      task_description = 'Q: Is vertex %s connected to vertex %s? List the answers after "Ans:" in the format of [Yes, No,].\nA: ' % (
            name_dict[source],
            name_dict[target],
        )
      question += task_description
      answer = 'No,'
      for edge in graph.e[0]:
        if source in edge and target in edge:
            answer = 'Yes,'
            break
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [source, target],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    source, target = random.sample(list(graph.v), k=2)
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    task_description = 'Q: Is vertex %s connected to vertex %s? List the answers after "Ans:" in the format of [Yes, No,].\nA: ' % (
            name_dict[source],
            name_dict[target],
        )
    question += task_description
    answer = 'Ans:[No,].'
    for edge in graph.e[0]:
      if source in edge and target in edge:
          answer = 'Ans:[Yes,].'
          break
    if 'Yes' in answer and cot:
      answer += (
            ' Because, vertex %s and %s are connected by a hyperedge in the hypergraph description.'
            % (name_dict[source], name_dict[target])
        )
    elif 'No' in answer and cot:
      answer += (
            ' Because, vertex %s and %s are not connected by any hyperedge in the hypergraph description.'
            % (name_dict[source], name_dict[target])
        )
    return question + answer


class VertexCount(GraphTask):
  """The hypergraph task for finding number of vertices in a hypergraph."""
  def __init__(self):
    super().__init__()
    self.name = 'vertex_count'
    self._task_description = 'Q: How many vertices are in this hypergraph? list the answers after "Ans" in the format like [10].\nA: '

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      question += self._task_description
      answer = ' %d.' % len(graph.v)
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': self._task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [],
      }
    return examples_dict

  def get_vertices_string(self, name_dict, nvertices):
    vertex_string = ''
    for i in range(nvertices - 1):
      vertex_string += name_dict[i] + ', '
    vertex_string += 'and ' + name_dict[nvertices - 1]
    return vertex_string

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    question += self._task_description
    answer = 'Ans:[%d].' % len(graph.v)
    if cot:
      answer += ' The vertices are %s.' % self.get_vertices_string(
          name_dict, len(graph.v)
      )

    return question + answer


class VertexDegree(GraphTask):
  """The hypergraph task for finding degree of a vertex in a hypergraph."""

  def __init__(self):
    super().__init__()
    self.name = 'vertex_degree'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      source_vertex = random.sample(list(graph.v), k=1)[0]
      task_description = (
          'Q: What is the degree of vertex %s? list the answers after "Ans" in the format like [10].\nA: ' % name_dict[source_vertex]
      )
      question += task_description
      degree = 0 
      for edge in graph.e[0]:
        if source_vertex in edge:
          degree += 1 
      
      answer = '%d.' % degree
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [source_vertex],
      }
    return examples_dict

  def get_edge_string(
      self, name_dict, graph, source_vertex
  ):
    """Gets a string identifying the edges a given vertex is connected to."""
    edge_string = ''
    target_edges = graph.edges(source_vertex)
    if target_edges:
      for i in range(len(target_edges)):
        edge_string += str(name_dict[target_edges[i]]) + ','
      edge_string = edge_string[:-1]
    else:
      edge_string = 'no hyperedges'
    return edge_string
  
  def get_edge_vertices_string(
      self, name_dict, graph, source_vertex
  ):
    """Gets a string identifying the edges a given vertex is connected to."""
    edge_string = ''
    target_edges = graph.edges(source_vertex)
    target_edges = [graph.e[0][i] for i in target_edges]

    if target_edges:
      for edge in target_edges:
        tmp = ''
        for i in edge:
          tmp += "%s," % (name_dict[i])
        tmp = '(' + tmp[:-1] + '),'
        edge_string += tmp 
    else:
      edge_string = 'no hyperedges'
    return edge_string
  def get_star_expanation_string(self,name_dict,graph,source_vertex):
    neighbor = graph.clique_neighbor(source_vertex)
    tmp = ''
    for vertex in neighbor:
      tmp += f'{name_dict[vertex]},'
    return tmp[:-1]

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    edge_dict = hypergraph_text_encoder.EDGES_ENCODER_DICT[encoding_method]
    source_vertex = random.sample(list(graph.v), k=1)[0]
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    
    question += (
        'Q: What is the degree of vertex %s? list the answers after "Ans" in the format like [10].\nA: ' % name_dict[source_vertex]
    )
    degree = 0 
    for edge in graph.e[0]:
      if source_vertex in edge:
        degree += 1 
    answer = 'Ans:[%d].' % degree
    if cot:
      if degree != 0:
        answer += ' This is because vertex %s is connected to hyperedges %s.' % (
            name_dict[source_vertex],
            self.get_edge_string(edge_dict, graph, source_vertex),
        )
      else:
        answer += ' This is because vertex %s is not connected to any hyperedges.' % (
              name_dict[source_vertex],
          )
    return question + answer



class HyperedgeDegree(GraphTask):
  """The graph task for finding degree of a vertex in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'hyperedge_degree'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    edge_dict = hypergraph_text_encoder.EDGES_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      source_edge = random.sample(list(range(len(graph.e[0]))), k=1)[0]
      task_description = (
          'Q: What is the degree of hyperedge %s? list the answers after "Ans" in the format like [10].\nA: ' % self.get_edge_string(name_dict,edge_dict,graph,source_edge,encoding_method)
      )
      question += task_description
      degree = 0 
      for vertex in graph.e[0][source_edge]:
        degree += 1 
      answer = '%d.' % degree
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [source_edge],
      }
    return examples_dict

  def get_edge_string(
      self, name_dict, edge_dict,graph, source_edge,encoding_method
  ):
    """Gets a string identifying the edges a given vertex is connected to."""
    return f'{edge_dict[source_edge]}'
  
  def get_star_expanation_string(self,name_dict,graph,source_vertex):
    neighbor = graph.clique_neighbor(source_vertex)
    tmp = ''
    for vertex in neighbor:
      tmp += f'{name_dict[vertex]},'
    return tmp[:-1]

  def get_edge_vertices_string(self,name_dict, graph, source_edge):
    edges = graph.e[0][source_edge]
    tmp = ''
    for i in edges:
      tmp += f'{name_dict[i]},'
    tmp = '(' + tmp[:-1] + ')'
    return tmp

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    edge_dict = hypergraph_text_encoder.EDGES_ENCODER_DICT[encoding_method]
    source_edge = random.sample(list(range(len(graph.e[0]))), k=1)[0]
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    task_description = (
          'Q: What is the degree of hyperedge %s? list the answers after "Ans" in the format like [10].\nA: ' % self.get_edge_string(name_dict,edge_dict,graph,source_edge,encoding_method)
    )
    question += task_description
    degree = 0 
    for vertex in graph.e[0][source_edge]:
        degree += 1 
    answer = 'Ans:[%d].' % degree
    if cot:
      if degree != 0:
        answer += ' This is because hyperedge %s is connected to vertices %s.' % (
              self.get_edge_string(name_dict,edge_dict,graph,source_edge,encoding_method),
              self.get_edge_vertices_string(name_dict, graph, source_edge)[1:-1],
          )
      else:
        answer += ' This is because hyperedge %s is not connected to any vertices.' % (
              self.get_edge_string(name_dict,edge_dict,graph,source_edge,encoding_method),
          )
    return question + answer


class HyperedgeCount(GraphTask):
  """The graph task for finding number of edges in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'hyperedge_count'
    self._task_description = 'Q: How many hyperedges are in this hypergraph? list the answers after "Ans" in the format like [10].\nA: '

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      question += self._task_description
      answer = ' %d.' % len(graph.e[0])
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': self._task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [],
      }
    return examples_dict

  def get_edge_vertices_string(
      self, name_dict, graph
  ):
    """Gets a string identifying the edges a given vertex is connected to."""
    edge_string = ''
    target_edges = graph['hypedges']

    if target_edges:
      for edge in graph.e[0]:
        tmp = ''
        for i in edge:
          tmp += "%s," % (name_dict[i])
        tmp = '(' + tmp[:-1] + '),'
        edge_string += tmp 
    else:
      edge_string = 'no hyperedges'
    return edge_string
  
  def get_edges_string(
      self, name_dict, graph
  ):
    """Gets a string identifying the edges a given vertex is connected to."""
    edge_string = ''
    target_edges = graph['hypedges']
    if target_edges:
      for i in range(len(target_edges)-1):
        edge_string += str(name_dict[i]) + ', '
      edge_string += 'and ' + str(name_dict[len(target_edges)-1])
    else:
      edge_string = 'no hyperedges'
    return edge_string

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    edge_dict = hypergraph_text_encoder.EDGES_ENCODER_DICT[encoding_method]
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    question += self._task_description
    answer = 'Ans:[%d].' % len(graph.e[0])
    if cot:
      answer += ' The hyperedges are %s.' % self.get_edges_string(
            edge_dict, graph
        )
    return question + answer


class ConnectedVertices(GraphTask):
  """The graph task for finding connected vertices to a given vertex in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'connected_vertices'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      source_vertex = random.sample(list(graph.v), k=1)[0]
      task_description = f'Q: List all the vertices connected to {name_dict[source_vertex]} in alphabetical order. List all the answers after "Ans" in the format of [{name_dict[0]},{name_dict[1]},{name_dict[2]}] and separate the answers by a comma.\nA: '
      question += task_description
      graph = HyperGraph(graph.v,graph.e[0])
      outgoing_edges = list(graph.edges(source_vertex))
      outgoing_edges = [graph.e[0][i] for i in outgoing_edges]
      if outgoing_edges:
        answer = self.get_connected_vertices(source_vertex,outgoing_edges, name_dict) + '.'
      else:
        answer = 'No vertices.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [source_vertex],
      }
    return examples_dict

  def get_connected_vertices(
      self, source_vertex,edges, name_dict
  ):
    """Gets a string including all the vertices that are connected to source."""
    connected_vertices = []
    for edge in edges:
      for i in edge:
        if i not in connected_vertices and i != source_vertex:
          connected_vertices.append(i) 
    # remove repeated vertices 
    connected_vertices = sorted(connected_vertices)
    connected_vertices = [name_dict[i] for i in connected_vertices]
    connected_vertices_string = ''
    if connected_vertices:
      try:
        int(connected_vertices[0])
        connected_vertices_string = ','.join(map(str, connected_vertices))
      except ValueError:
        connected_vertices_string = ','.join(map(str, connected_vertices))
    return connected_vertices_string

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    source_vertex = random.sample(list(graph.v), k=1)[0]
    task_description = f'Q: List all the vertices connected to {name_dict[source_vertex]} in alphabetical order. List all the answers after "Ans" in the format of [{name_dict[0]},{name_dict[1]},{name_dict[2]}] and separate the answers by a comma.\nA: '
    question += task_description
    outgoing_edges = list(graph.edges(source_vertex))
    outgoing_edges = [graph.e[0][i] for i in outgoing_edges]
    answer = ''
    edge_name = 'hyperedges'
    if outgoing_edges:
      answer = "Ans:[" + self.get_connected_vertices(source_vertex,outgoing_edges, name_dict) + '].'
      if cot:
          answer += ' This is because there is %s connecting %s to %s,' % (
                edge_name,
                name_dict[source_vertex],
                self.get_connected_vertices(source_vertex,outgoing_edges, name_dict).split(','),
            )
    else:
      answer = 'Ans:[].'
      if cot:
        answer += (
              ' This is because %s is not connected to any vertices through %s.'
              % (name_dict[source_vertex],edge_name)
          )
    return question + answer


class DisconnectedVertices(GraphTask):
  """The task for finding disconnected vertices for a given vertex in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'disconnected_vertices'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      source_vertex = random.sample(list(graph.v), k=1)[0]
      task_description = f'Q: List all the vertices that are not connected to {name_dict[source_vertex]} in alphabetical order. List all the answers after "Ans" in the format of [{name_dict[0]},{name_dict[1]},{name_dict[2]}] and separate the answers by a comma.\nA: '
      question += task_description
      graph = HyperGraph(graph.v,graph.e[0])
      outgoing_edges = list(graph.edges(source_vertex))
      outgoing_edges = [graph.e[0][i] for i in outgoing_edges]
      answer = self.get_disconnected_vertices(
          source_vertex, outgoing_edges, name_dict, list(graph.v)
      )
      if not answer:
        answer = 'No vertices'

      answer += '.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [source_vertex],
      }
    return examples_dict

  def get_disconnected_vertices(
      self,
      source,
      edges,
      name_dict,
      all_vertices,
  ):
    """Gets a string with all the vertices that are not connected to source."""
    for edge in edges:
      for vertex in edge:
        if vertex in all_vertices:
          all_vertices.remove(vertex)
        
    if source in all_vertices:
      all_vertices.remove(source)
    
    all_vertices_names = []
    for vertex in all_vertices:
      all_vertices_names.append(name_dict[vertex])
    if all_vertices_names:
      try:
        int(all_vertices_names[0])
        for ind, value in enumerate(all_vertices_names):
          all_vertices_names[ind] = int(value)
        all_vertices_names = all_vertices_names
        for ind, value in enumerate(all_vertices_names):
          all_vertices_names[ind] = str(value)
      except ValueError:
        pass
    return ','.join(map(str, all_vertices_names))

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    source_vertex = random.sample(list(graph.v), k=1)[0]
    task_description = f'Q: List all the vertices that are not connected to {name_dict[source_vertex]} in alphabetical order. List all the answers after "Ans" in the format of [{name_dict[0]},{name_dict[1]},{name_dict[2]}] and separate the answers by a comma.\nA: '
    question += task_description
    outgoing_edges = list(graph.edges(source_vertex))
    outgoing_edges = [graph.e[0][i] for i in outgoing_edges]
    answer = ''
    disconnected_vertices_string = self.get_disconnected_vertices(
        source_vertex, outgoing_edges, name_dict, list(graph.v)
    )
    edge_name = 'hyperedges'
    if disconnected_vertices_string:
      answer = "Ans:[" + disconnected_vertices_string + '].'
      if cot:
        answer += ' This is because'
        answer += ' there is not %s connecting %s to %s,' % (
                edge_name,
                name_dict[source_vertex],
                disconnected_vertices_string,
            )
    else:
      answer = 'Ans:[].'
      if cot:
        answer += (
              ' This is because the vertex %s is connected to all the vertices through %s.'
              % (name_dict[source_vertex],edge_name)
          )
    return question + answer


class ReachabilityCheck(GraphTask):
  """The graph task to check if there is a path from a source to target."""

  def __init__(self):
    super().__init__()
    self.name = 'reachability'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.v), k=2)
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      task_description = 'Q: Is there a path from vertex %s to vertex %s? List the answers after "Ans:" in the format of [Yes, No,].\nA: ' % (
          name_dict[source],
          name_dict[target],
      )
      question += task_description
      if graph.has_path(source,target):
        answer = 'Yes,'
      else:
        answer = 'No,'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [source, target],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    source, target = random.sample(list(graph.v), k=2)
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    question += 'Q: Is there a path from vertex %s to vertex %s? List the answers after "Ans:" in the format of [Yes, No,].\nA: ' % (
        name_dict[source],
        name_dict[target],
    )
    
    edge_name = 'hyperedge'
    graph_name = 'hypergraph'
    if graph.has_path(source,target):
      answer = 'Ans:[Yes,].'
      if cot:
        path = graph.short_path(source,target)
        explanation = ' Because'
        for i in range(len(path) - 1):
          if len(path) == 2 or i < len(path) - 2:
            sep = ','
          else:
            sep = ', and'
          explanation += '%s there is a %s connecting vertex %s to vertex %s' % (
              sep,
              edge_name,
              name_dict[path[i][0]],
              name_dict[path[i + 1][0]],
          )
        explanation += ' .'
        answer += explanation
    else:
      answer = 'Ans:[No,].'
      if cot:
        answer += (
            ' Because, there is no path connecting vertex %s to vertex %s based on'
            ' the %s description.' % (name_dict[source], name_dict[target],graph_name)
        )
    return question + answer


class ShortestPath(GraphTask):
  """The graph task to check if there is a path from a source to target."""

  def __init__(self):
    super().__init__()
    self.name = 'shortest_path'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.v), k=2)
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      task_description = (
          'Q: What is the length of the shortest path from vertex %s to vertex'
          ' %s? List the answers after "Ans:" in the format like [10].\nA: '
          % (
              name_dict[source],
              name_dict[target],
          )
      )
      question += task_description
      try:
        path = graph.short_path(source, target)
        if path is None:
            raise nx.NetworkXNoPath
        answer = str(len(path) - 1) + '.'
      except nx.NetworkXNoPath:
        answer = 'There is no path from vertex %s to vertex %s.' % (
            name_dict[source],
            name_dict[target],
        )
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [source, target],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    source, target = random.sample(list(graph.v), k=2)
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    question += (
        'Q: What is the length of the shortest path from vertex %s to vertex'
        ' %s? List the answers after "Ans:" in the format like [10].\nA: '
        % (
            name_dict[source],
            name_dict[target],
        )
    )
    edge_name = 'hyperedge'
    graph_name = 'hypergraph' 
    if graph.has_path(source, target):
      # path = nx.shortest_path(graph, source, target)
      path = graph.short_path(source, target)
      answer = 'Ans:['+str(len(path) - 1) + '].'
      if cot:
        explanation = ' Because'
        for i in range(len(path) - 1):
          if len(path) == 2 or i < len(path) - 2:
            sep = ','
          else:
            sep = ', and'
          explanation += '%s there is a %s connecting vertex %s to vertex %s' % (
              sep,
              edge_name,
              name_dict[path[i][0]],
              name_dict[path[i + 1][0]],
          )
        explanation += ' .'
        answer += explanation
    else:
      answer = 'Ans:[No path].'
      if cot:
        answer += (
            ' Because, there is no path connecting vertex %s to vertex %s based on'
            ' the %s description.' % (name_dict[source], name_dict[target],graph_name)
        )
    return question + answer

import dhg


class VertexSetConnectionCheck(GraphTask):
  """The hypergraph task to check if set A connected to set B"""

  def __init__(self):
    super().__init__()
    self.name = 'vertexset_connection'

  def get_vertex_set_string(self,name_dict,list):
    result = ''
    for i in list:
      result += name_dict[i] + ','
    result = '(' + result[:-1] + ')'
    return result

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      # produce postive instance 
      p = random.random()
      hyperedges = graph.e[0]
      if p > 0.5 or len(hyperedges)<=1: 
        times = 0
        while True:
          selected_edge = random.choice(hyperedges)
          if len(selected_edge) == 2 and times < len(hyperedges): 
            times += 1
            continue 
          break
        len_list1 = random.randint(1, len(selected_edge)-1)
        list1 = random.sample(selected_edge, len_list1)
        list2 = [item for item in selected_edge if item not in list1]
        begin = 1 
        if len(list1) == 1 and len(selected_edge)>2: 
          begin = 2 
        len_list2 = random.randint(begin, len(list2))
        list2 = random.sample(list2, len_list2)
        answer = 'Yes.'
      else:
        while True:
          edge1 , edge2 = random.sample(hyperedges,2)
          common = list(set(edge1).intersection(set(edge2)))
          len_list1 = random.randint(2, len(edge1))
          list1 = random.sample(edge1, len_list1)
          edge2 = [item for item in edge2 if item not in list1]
          if len(edge2) == 0 : 
            continue
          len_list2 = random.randint(1, len(edge2))
          list2 = random.sample(edge2, len_list2)
          if set(list1).issubset(set(common)) and set(list2).issubset(set(common)):
            continue
          answer = 'No.'
          for edge in hyperedges:
            if set(list1 + list2).issubset(edge):
              answer = 'Yes.'
              print("change")
          break

      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      
      task_description = 'Q: Is there a hyperedge that contain both vertex set %s and vertex set %s? List the answers after "Ans:" in the format of [Yes, No,].\nA: ' % (
          self.get_vertex_set_string(name_dict,list1),
          self.get_vertex_set_string(name_dict,list2),
      )
      question += task_description
      
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [list1, list2],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    p = random.random()
    hyperedges = graph.e[0]
    if p > 0.5 or len(hyperedges)<=1: 
        # 
      times = 0
      while True:
        selected_edge = random.choice(hyperedges)
        if len(selected_edge) == 2 and times < len(hyperedges): 
          times += 1
          continue 
        break
      len_list1 = random.randint(1, len(selected_edge)-1)
      list1 = random.sample(selected_edge, len_list1)
      list2 = [item for item in selected_edge if item not in list1]
      begin = 1 
      if len(list1) == 1 and len(selected_edge)>2: 
        begin = 2 
      len_list2 = random.randint(begin, len(list2))
      list2 = random.sample(list2, len_list2)
      answer = 'Ans:[Yes,]'
    else:
      while True:
        edge1 , edge2 = random.sample(hyperedges,2)
        common = list(set(edge1).intersection(set(edge2)))
        len_list1 = random.randint(2, len(edge1))
        list1 = random.sample(edge1, len_list1)
        edge2 = [item for item in edge2 if item not in list1]
        if len(edge2) == 0 : 
          continue
        len_list2 = random.randint(1, len(edge2))
        list2 = random.sample(edge2, len_list2)
        if set(list1).issubset(set(common)) and set(list2).issubset(set(common)):
          continue
        answer = 'Ans:[No,]'
        for edge in hyperedges:
          if set(list1 + list2).issubset(edge):
              answer = 'Ans:[Yes,]'
              print("change")
        break

    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
  
    task_description = 'Q: Is there a hyperedge that contain both vertex set %s and vertex set %s? List the answers after "Ans:" in the format of [Yes, No,].\nA: ' % (
        self.get_vertex_set_string(name_dict,list1),
        self.get_vertex_set_string(name_dict,list2),
    )
    
    question += task_description

    if 'Yes' in answer and cot:
      answer += (
            ' Because, there is a hyperedge that contains both vertex set %s and vertex set %s.'
            % (self.get_vertex_set_string(name_dict,list1), self.get_vertex_set_string(name_dict,list2))
        )
    elif 'No' in answer and cot: 
      answer += (
            ' Because, there is no hyperedge that contains both vertex set %s and vertex set %s.'
            % (self.get_vertex_set_string(name_dict,list1), self.get_vertex_set_string(name_dict,list2))
        )
     

    return question + answer



class VertexSet_In_HyperedgeCheck(GraphTask):
  """The hypergraph task to check if set A is connected by a hyperedge"""

  def __init__(self):
    super().__init__()
    self.name = 'vertexsetin_hyperedge'

  def get_vertex_set_string(self,name_dict,list):
    result = ''
    for i in list:
      result += name_dict[i] + ','
    result = '(' + result[:-1] + ')'
    return result
  
  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      p = random.random()
      if p > 0.6:
        hyperedges = graph.e[0]
        selected_edge = random.choice(hyperedges)
        if len(selected_edge) >= 3: 
          min_len = 3 
        else:
          min_len = len(selected_edge)
        len_set = random.randint(min_len, len(selected_edge))
        vertex_set = random.sample(selected_edge, len_set)
      else:
        num_v = len(graph.v)
        deg_e_list = list(range(2, num_v + 1))
        prob_k_list = [3 ** (-k) for k in range(len(deg_e_list))]
        sum_of_prob_k_list = sum(prob_k_list)
        prob_k_list = [prob_k / sum_of_prob_k_list for prob_k in prob_k_list]
        k = random.choices(deg_e_list, weights=prob_k_list)[0]
        e = random.sample(range(num_v), k)
        e = tuple(sorted(e))
        vertex_set = e
      
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      
      task_description = 'Q: Is there a hyperedge that contain all vertices in vertex set %s? List the answers after "Ans:" in the format of [Yes, No,].\nA: ' % (
        self.get_vertex_set_string(name_dict,vertex_set)
      )
      question += task_description
      answer = 'No.'
      for edge in graph.e[0]:
        if set(vertex_set).issubset(set(edge)):
          answer = 'Yes.'
          break
        
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': vertex_set,
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    p = random.random()
    if p > 0.6:
      hyperedges = graph.e[0]
      selected_edge = random.choice(hyperedges)
      if len(selected_edge) >= 3: 
        min_len = 3 
      else:
        min_len = len(selected_edge)
      len_set = random.randint(min_len, len(selected_edge))
      vertex_set = random.sample(selected_edge, len_set)
    else:
      num_v = len(graph.v)
      deg_e_list = list(range(2, num_v + 1))
      prob_k_list = [3 ** (-k) for k in range(len(deg_e_list))]
      sum_of_prob_k_list = sum(prob_k_list)
      prob_k_list = [prob_k / sum_of_prob_k_list for prob_k in prob_k_list]
      k = random.choices(deg_e_list, weights=prob_k_list)[0]
      e = random.sample(range(num_v), k)
      e = tuple(sorted(e))
      vertex_set = e
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    task_description = 'Q: Is there a hyperedge that contain all vertices in vertex set %s? List the answers after "Ans:" in the format of [Yes, No,].\nA: ' % (
        self.get_vertex_set_string(name_dict,vertex_set)
    )
    question += task_description
    answer = 'Ans:[No,]'
    for edge in graph.e[0]:
      if set(vertex_set).issubset(set(edge)):
        answer = 'Ans:[Yes,]'
    # 
    if 'Yes' in answer and cot:
      answer += (
            ' Because, there exists a hyperedge that contains all vertices in vertex set %s.'
            % (self.get_vertex_set_string(name_dict,vertex_set))
        )
    elif 'No' in answer and cot: 
      answer += (
            ' Because, there is no hyperedge that contains all vertices in vertex set %s.'
            % (self.get_vertex_set_string(name_dict,vertex_set))
        )
    return question + answer
  
class Hyperedge_In_HyperedgeCheck(GraphTask):
  """The graph task to judge whether there is an inclusion relation between hyperedges"""

  def __init__(self):
    super().__init__()
    self.name = 'hyperedge_inhyperedge'

  def prepare_examples_dict(
          self,
          graphs,
          generator_algorithms,
          encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      task_description = (
              'Q: Whether any hyperedge in the hypergraph is contained by other hyperedges ?'
      )
      question += task_description
      result = self.get_inclusion(graph, name_dict)
      if result != {}:
        answer = 'Yes,'
      else:
        answer = 'No,'
      examples_dict[ind] = {
        'question': question,
        'answer': answer,
        'nvertices': str(len(graph.v)),
        'nedges': str(len(graph.e[0])),
        'task_description': task_description,
        'graph': graph,
        'algorithm': generator_algorithms[ind],
        'vertex_ids': [],
      }
    return examples_dict

  def get_inclusion(
          self, graph, name_dict
  ):
    # 
    sorted_edge = sorted(graph.e[0], key=len)
    # 
    result = {}
    for i in range(len(sorted_edge)-1):
      # 
      j = i+1
      while j < len(sorted_edge):
        # 
        if set(sorted_edge[i]) <= set(sorted_edge[j]):
          flag = 1
          if i not in result:
            flag = 0
            tmp = ''
            for v in sorted_edge[i]:
              tmp += "%s," % (name_dict[v])
            result[i] = ' (' + tmp[:-1] + ') is included in '
          tmp = ''
          for v in sorted_edge[j]:
            tmp += "%s," % (name_dict[v])
          tmp = '(' + tmp[:-1] + ').'
          if flag == 1:  # 
            result[i] = result[i][:-1] + ', ' + tmp
          else: # 
            result[i] += tmp
        j+=1
    # print("result",result)
    return result

  def create_few_shot_example(
          self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    question += (
            'Q: Whether any hyperedge in the hypergraph is contained by other hyperedges?'
            ' List the answers after "Ans:" in the format like [Yes, No,].\nA: '
    )
    result = self.get_inclusion(graph,name_dict)
    if result != {}:
      answer = 'Ans:[Yes,].'
      if cot:
        explanation = ' Because there is the following inclusion relation in the hyperedges after analysis:'
        if len(result) > 1:
          explanation = ' Because there are the following inclusion relations in the hyperedges after analysis:'
        for value in result.values():
          explanation += value
        answer += explanation
    else:
      answer = 'Ans:[No,].'
      if cot:
        explanation = ' Because after analysis, no inclusion relation in the hyperedges on the hypergraph description.'
        answer += explanation
    return question + answer


# NOTE: 
class SharedVerticesBetweenHyperedges(GraphTask):
  """The hypergraph task for finding the common vertices between two hyperedges."""

  def __init__(self):
    super().__init__()
    self.name = 'sharedvertices'
    
  def get_edge_string(
      self, name_dict, edge_dict,graph, source_edge,encoding_method
  ):
    """Gets a string identifying the edges a given vertex is connected to."""
    if 'incident_edge' in encoding_method:
      edge_string = ''
      for i in graph.e[0][source_edge]:
        edge_string += f'{name_dict[i]},'
      edge_string = '(' + edge_string[:-1] + ')'
      return edge_string
    else:
      return f'{edge_dict[source_edge]}'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    edge_dict = hypergraph_text_encoder.EDGES_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      if len(graph.e[0]) < 2: 
        continue
      source_edge , target_edge = random.sample(list(range(len(graph.e[0]))),k=2)
      task_description = f'Q: List the vertices connected to both hyperedge {self.get_edge_string(name_dict,edge_dict,graph,source_edge,encoding_method)} and hyperedge {self.get_edge_string(name_dict,edge_dict,graph,target_edge,encoding_method)} in alphabetical order. List all the answers after "Ans" in the format of [{name_dict[0]},{name_dict[1]},{name_dict[2]}] and separate the answers by a comma.\nA: '
      question += task_description
      source_edge_vertices = graph.e[0][source_edge]
      target_edge_vertices = graph.e[0][target_edge]
      answer =  list(set(source_edge_vertices)&set(target_edge_vertices))
      if len(answer) != 0:
        answer = [name_dict[i] for i in answer]
        answer = ",".join(answer)
      else:
        answer = 'No vertices.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': task_description,
          'graph': graph,
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [source_edge,target_edge],
      }
    return examples_dict
  
  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    if len(graph.e[0]) < 2: 
        number_of_vertices = random.choice(range(5,10))
        number_of_hypedges = random.choice(range(2,int(number_of_vertices*1.5)))
        graph = dhg.random.hypergraph_Gnm(num_v=int(number_of_vertices),num_e=number_of_hypedges)
        graph = HyperGraph(graph.v,graph.e[0])
      
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    edge_dict = hypergraph_text_encoder.EDGES_ENCODER_DICT[encoding_method]
    question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
    source_edge , target_edge = random.sample(list(range(len(graph.e[0]))),k=2)
    question += f'Q: List the vertices connected to both hyperedge {self.get_edge_string(name_dict,edge_dict,graph,source_edge,encoding_method)} and hyperedge {self.get_edge_string(name_dict,edge_dict,graph,target_edge,encoding_method)} in alphabetical order. List all the answers after "Ans" in the format of [{name_dict[0]},{name_dict[1]},{name_dict[2]}] and separate the answers by a comma.\nA: '
    source_edge_vertices = graph.e[0][source_edge]
    target_edge_vertices = graph.e[0][target_edge]
    answer =  list(set(source_edge_vertices)&set(target_edge_vertices))
    if len(answer) != 0:
      answer = [name_dict[i] for i in answer]
      answer = ",".join(answer)
    else:
      answer = 'No vertices.'
    edge_name = 'hyperedges'

    if answer != 'No vertices.':
      answer = "Ans:[" + answer + '].'
      if cot:
        answer += ' This is because vertices %s are both in hyperedge %s and hyperedge %s.' % (
            answer,
            self.get_edge_string(name_dict,edge_dict,graph,source_edge,encoding_method),
            self.get_edge_string(name_dict,edge_dict,graph,target_edge,encoding_method),
        )
    else:
      answer = 'Ans:[].'
      if cot:
        answer += (
            ' This is because hyperedge %s and hyperedge %s have no common connecting vertices.'% (
            self.get_edge_string(name_dict,edge_dict,graph,source_edge,encoding_method),
            self.get_edge_string(name_dict,edge_dict,graph,target_edge,encoding_method),
          )
        )
    return question + answer

# NOTE: 
class IsomorphismRecognition(GraphTask):
  """The graph task for test if two hypergraph are Isomorphism."""

  def __init__(self):
    super().__init__()
    self.name = 'graph_isomorphism'
    self._task_description = 'Q: Are these two hypergraphs isomorphism? list the answers after "Ans" in the format of [Yes, No,].\nA: '

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):

      if random.random() > 0.5:
        # create Isomorphism
        graph_text1 = hypergraph_text_encoder.encode_graph(graph, encoding_method)
        graph_shuf = graph.shuffleNode()
        graph_text2 = hypergraph_text_encoder.encode_graph(graph_shuf, encoding_method)
        answer = 'Yes.'
      else:
        # create non-Isomorphism
        graph_text1 = hypergraph_text_encoder.encode_graph(graph, encoding_method)
        num_vertices = len(graph.v)
        edge_degree = [len(e) for e in graph.e[0]]
        num_e = len(edge_degree)
        edges = set()
        while len(edges) < num_e:
            k = edge_degree[len(edges)]
            e = random.sample(range(num_vertices), k)
            e = tuple(sorted(e))
            if e not in edges:
                edges.add(e)
        graph_shuf = HyperGraph(list(range(len(graph.v))),list(edges))
        graph_text2 = hypergraph_text_encoder.encode_graph(graph_shuf, encoding_method)
        answer = 'No.'
      
      #  check 
      from test_isomo import HGSCKernel
      import matplotlib.pyplot as plt
      model=HGSCKernel()
      graph = dhg.Hypergraph(len(graph.v),graph.e[0])
      graph_shuf = dhg.Hypergraph(len(graph_shuf.v),graph_shuf.e[0])
      if model.test_isomo(graph,graph_shuf):
        if answer == 'No.':
          print("no is wrong")
        answer = 'Yes.'
      else:
        if answer == 'Yes.':
          print("yes is wrong")
        answer = 'No.'
      graph_text1 = graph_text1.replace('G','H')
      question = "There are two hypergraphs: H and G.\nThe description of H is: " + graph_text1 + 'The description of G is: '+graph_text2 + self._task_description
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nvertices': str(len(graph.v)),
          'nedges': str(len(graph.e[0])),
          'task_description': self._task_description,
          'graph': [graph,graph_shuf],
          'algorithm': generator_algorithms[ind],
          'vertex_ids': [],
      }
    return examples_dict

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
    edge_dict = hypergraph_text_encoder.EDGES_ENCODER_DICT[encoding_method]
    if random.random() > 0.5:
        # create Isomorphism
        graph_text1 = hypergraph_text_encoder.encode_graph(graph, encoding_method)
        graph_shuf = graph.shuffleNode()
        graph_text2 = hypergraph_text_encoder.encode_graph(graph_shuf, encoding_method)
        answer = 'Yes.'
    else:
      # create non-Isomorphism
      graph_text1 = hypergraph_text_encoder.encode_graph(graph, encoding_method)
      num_vertices = len(graph.v)
      edge_degree = [len(e) for e in graph.e[0]]
      num_e = len(edge_degree)
      edges = set()
      while len(edges) < num_e:
          k = edge_degree[len(edges)]
          e = random.sample(range(num_vertices), k)
          e = tuple(sorted(e))
          if e not in edges:
              edges.add(e)
      graph_shuf = HyperGraph(graph.v,list(edges))
      graph_text2 = hypergraph_text_encoder.encode_graph(graph_shuf, encoding_method)
      from test_isomo import HGSCKernel
      model=HGSCKernel()
      if model.test_isomo(graph,graph_shuf):
        answer = 'Yes.'
      else:
        answer = 'No.'
    graph_text1 = graph_text1.replace('G', 'H')
    question = "There are two hypergraphs: H and G.\nThe description of H is: " + graph_text1 + 'The description of G is: '+graph_text2 + self._task_description
    if cot:
      if answer =='Yes.':
        answer += (
          "In the incidence matrix of a hypergraph, rows represent vertices and columns represent hyperedges. "
          "The incidence matrix1 of H is: " + self.get_adj_matrix(graph) + ", and the incidence matrix2 of G is: " + self.get_adj_matrix(graph_shuf) + ". "
          "Do row or column swaps on incidence matrix1 step by step, and make sure to record every change in the incidence matrix1 accurately, the final incidence matrix 1 can be changed into incidence matrix 2. "
          "Just as swapping rows 1 and 2, or swapping columns 1 and 2 in matrix [[0,1][1,0]], the result is [[1,0][0,1]], so H is isomorphic to G.\n"
        )
      else:
        answer += (
          "The incidence matrix of a hypergraph represents the relationship between hyperedges and vertices. "
          "The incidence matrix1 of H is: " + self.get_adj_matrix(graph) + ", and the incidence matrix2 of G is: " + self.get_adj_matrix(graph_shuf) + ". "
          "Do row or column swaps on incidence matrix1 step by step, and make sure to record every change in the incidence matrix1 accurately, the final incidence matrix 1 can never be changed into incidence matrix 2. "
          "Just as swapping rows 1 and 2, or swapping columns 1 and 2 in matrix [[0,1][1,0]], the result is [[1,0][0,1]] but never [[0,0][1,1]], so H is not isomorphic to G.\n"
        )
    return question + answer

  def get_adj_matrix(self,hypergraph):
    adj_matrix_str = '[,'
    for edge in hypergraph.e[0]:
      list = ['0'] * len(hypergraph.v)
      for vertex in edge:
        list[int(vertex)] = '1'
      edge_str = '['
      for i in list:
        edge_str += (i + ',')
      adj_matrix_str = adj_matrix_str[:-1] + edge_str[:-1] + '],'
    adj_matrix_str = adj_matrix_str[:-1] + ']'
    print(adj_matrix_str)
    return adj_matrix_str

  def get_com_label(self, count):
    com_label = '{1,'
    for i in range(count):
      com_label += '1'
    com_label+='}'
    return com_label

  def get_connected_vertices(
          self, source_vertex, edges, name_dict
  ):
    """Gets a string including all the vertices that are connected to source."""
    connected_vertices = []
    for edge in edges:
      for i in edge:
        if i not in connected_vertices and i != source_vertex:
          connected_vertices.append(i)
    connected_vertices = sorted(connected_vertices)
    connected_vertices = [name_dict[i] for i in connected_vertices]
    connected_vertices_string = ''
    if connected_vertices:
      try:
        int(connected_vertices[0])
        connected_vertices_string = ','.join(map(str, connected_vertices))
      except ValueError:
        connected_vertices_string = ','.join(map(str, connected_vertices))
    return len(connected_vertices), connected_vertices_string

