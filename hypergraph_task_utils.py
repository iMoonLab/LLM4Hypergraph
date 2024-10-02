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
import os
import random
import networkx as nx
from tensorflow.io import gfile
def create_example_feature(
    key,
    question,
    answer,
    algorithm,
    encoding_method,
    nvertices,
    nedges,
    graph = None
):
  """Create a tensorflow example from a datapoint."""
  key_feature = key
  question_feature = question

  answer_feature = answer
  algorithm_feature = algorithm
  encoding_method_feature = encoding_method
  nvertices_feature = nvertices
  nedges_feature = nedges

  example_feats= {
          'id': key_feature,
          'question': question_feature,
          'answer': answer_feature,
          'algorithm': algorithm_feature,
          'text_encoding': encoding_method_feature,
          'nvertices': nvertices_feature,
          'nedges': nedges_feature,
          'graph' : graph,
      }
  return example_feats


def load_graphs(
    base_path,
    algorithm,
    split,
    max_nvertices = 20,
):
  """Load a list of graphs from a given algorithm and split."""
  graphs_path = os.path.join(
      base_path,
      algorithm,
      split,
  )
  loaded_graphs = []
  all_files = gfile.listdir(graphs_path)
  for file in all_files:
    if file.endswith('.graphml'):
      path = os.path.join(graphs_path, file)
      graph = nx.read_graphml(open(path, 'rb'), node_type=int)
      if graph.number_of_vertices() <= max_nvertices:
        loaded_graphs.append(graph)
  return loaded_graphs



import pickle
from hyper_graph import HyperGraph
def load_hyper_graphs(
    base_path,
    algorithm,
    split,
    max_nvertices = 20,
):
  """Load a list of graphs from a given algorithm and split."""
  graphs_path = os.path.join(
      base_path,
      algorithm,
      split,
  )
  loaded_graphs = []
  all_files = gfile.listdir(graphs_path)
  for file in all_files:
    if file.endswith('.pkl'):
      path = os.path.join(graphs_path, file)
      with open(path,'rb') as f:
        graph = pickle.load(f)
      if len(graph.v) <= max_nvertices:
        graph = HyperGraph(graph.v,graph.e[0])
        loaded_graphs.append(graph)
  return loaded_graphs

def prepare_examples(
    examples_dict,
    encoding_method,
):
  """Create a list of tf.train.Example from a dict of examples."""
  examples = []
  for key, value in examples_dict.items():
    (
        question,
        answer,
        nvertices,
        nedges,
        algorithm,
        graph,
    ) = (
        value['question'],
        value['answer'],
        value['nvertices'],
        value['nedges'],
        value['algorithm'],
        value['graph'],
    )
    examples.append(
        create_example_feature(
            key,
            question,
            answer,
            algorithm,
            encoding_method,
            nvertices,
            nedges,
            graph
        )
    )
  return examples


def create_zero_shot_task(
    task,
    graphs,
    generator_algorithms,
    text_encoders,
    cot = False,
    prompt1='',
):
  """Create a recordio file with zero-shot examples for the task."""
  examples = []
  for encoding_method in text_encoders:
    examples_dict = task.prepare_examples_dict(
        graphs, generator_algorithms, encoding_method
    )
    if cot:
      for key in examples_dict.keys():
        examples_dict[key]['question'] += "Let's think step by step. "
    if prompt1 == 'v1':
      for key in examples_dict.keys():
        examples_dict[key]['question'] += "Let's think step by step. Make sure the data is calculated and recorded accurately at each step."
    elif prompt1 == 'v2':
      for key in examples_dict.keys():
        examples_dict[key]['question'] += "Let's analyze the connectivity by considering hyperedges linked to vertices and vertices linked through hyperedges."
    elif prompt1 == 'v3':
      for key in examples_dict.keys():
        examples_dict[key]['question'] += "Let's think hyperedges connected by vertices then vertices connected by hyperedges."
    examples += prepare_examples(examples_dict, encoding_method)
  return examples

import os 


def prepare_few_shots(
    task,
    graphs,
    text_encoders,
    cot,
):
  """Create a dict of few-shot examples with their cot for the task."""
  few_shots_examples_dict = {}
  for encoding_method in text_encoders:
    if encoding_method not in few_shots_examples_dict:
      few_shots_examples_dict[(encoding_method)] = []
    for graph in graphs:
      few_shots_examples_dict[(encoding_method)].append(
          task.create_few_shot_example(graph, encoding_method, cot)
      )
  return few_shots_examples_dict


def choose_few_shot_examples(
    few_shots_dict,
    encoding_method,
    k = 2,
):
  """Choose few shot examples for each algorithm."""
  few_shots_str = ''
  for _ in range(k):
    example_list = few_shots_dict[encoding_method]
    few_shots_str += 'Example: ' + random.choice(example_list) + '\n'
  return few_shots_str


def create_few_shot_task(
    task,
    graphs,
    generator_algorithms,
    few_shots_graphs,
    text_encoders,
    cot,
    bag,
    random_seed,
    prompt1='',
    one_shot = False,
):
  """Create a recordio file with few-shot examples for the task."""
  number_of_tokens = {}
  examples = []
  print('prepare few shot task', 'cot', cot, 'bag', bag)
  few_shots_examples_dict = prepare_few_shots(
      task,
      few_shots_graphs,
      text_encoders,
      cot,
  )
  for encoding_method in text_encoders:
    random.seed(random_seed)
    examples_dict = task.prepare_examples_dict(
        graphs, generator_algorithms, encoding_method
    )
    for key in examples_dict.keys():
      if not one_shot:
        few_shots_examples = choose_few_shot_examples(
            few_shots_examples_dict,
            encoding_method,
        )
      else:
        few_shots_examples = choose_few_shot_examples(
            few_shots_examples_dict,
            encoding_method,
            k=1
        )
      examples_dict[key]['question'] = (
          few_shots_examples + 'Example: ' + examples_dict[key]['question']
      )
      if bag:
          examples_dict[key]['question'] = examples_dict[key]['question'].replace(
              '\nQ: ',
              "\nLet's construct the hypergraph with the vertices and hyperedges first.\nQ: ",
          )
      
      if encoding_method not in number_of_tokens:
        number_of_tokens[encoding_method] = []
    examples += prepare_examples(examples_dict, encoding_method)

  return examples
