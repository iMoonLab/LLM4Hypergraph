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
r"""The graph tasks to be tried with LLMs."""
from collections.abc import Sequence
import os
import random
from absl import app
from absl import flags
import networkx as nx
import numpy as np
import pickle
import hypergraph_task
import hypergraph_task_utils as utils
import pandas as pd
_TASK = flags.DEFINE_enum(
    'task',
    None,
    [
        'HyperedgeCount',
        'VertexCount',
        'VertexDegree',
        'VertexConnectionCheck',
        'ReachabilityCheck',
        'ShortestPath',
        'ConnectedVertices',
        'DisconnectedVertices',
        'HyperedgeDegree',
        'VertexSetConnectionCheck',
        'VertexSet_In_HyperedgeCheck',
        'Hyperedge_In_HyperedgeCheck',
        'SharedVerticesBetweenHyperedges',
        'IsomorphismRecognition',
    ],
    'The task to generate datapoints.',
    required=True,
)
_ALGORITHM = flags.DEFINE_enum(
    'algorithm',
    None,
    ['hypergraph','hypergraph_high',''],
    'The graph generator algorithm to generate datapoints.',
    required=True,
)
_TASK_DIR = flags.DEFINE_string(
    'task_dir', None, 'The directory to write tasks.', required=True
)
_GRAPHS_DIR = flags.DEFINE_string(
    'graphs_dir', None, 'The directory containing the graphs.', required=True
)
_RANDOM_SEED = flags.DEFINE_integer(
    'random_seed',
    None,
    'The random seed to use for task generation.',
    required=True,
)


TASK_CLASS = {
    'HyperedgeCount':hypergraph_task.HyperedgeCount,
    'VertexCount':hypergraph_task.VertexCount,
    'VertexDegree':hypergraph_task.VertexDegree,
    'VertexConnectionCheck':hypergraph_task.VertexConnectionCheck,
    'ReachabilityCheck':hypergraph_task.ReachabilityCheck,
    'ShortestPath':hypergraph_task.ShortestPath,
    'ConnectedVertices':hypergraph_task.ConnectedVertices,
    'DisconnectedVertices':hypergraph_task.DisconnectedVertices,
    'HyperedgeDegree':hypergraph_task.HyperedgeDegree,
    'VertexSetConnectionCheck':hypergraph_task.VertexSetConnectionCheck,
    'VertexSet_In_HyperedgeCheck':hypergraph_task.VertexSet_In_HyperedgeCheck,
    'Hyperedge_In_HyperedgeCheck':hypergraph_task.Hyperedge_In_HyperedgeCheck,
    'SharedVerticesBetweenHyperedges':hypergraph_task.SharedVerticesBetweenHyperedges,
    'IsomorphismRecognition':hypergraph_task.IsomorphismRecognition,
}


def zero_shot(
    task,
    graphs,
    algorithms,
    text_encoders,
    cot,
    random_seed,
    split,
    prompt1='',
):
  """Creating zero-shot or zero-cot examples for the given task.

  Args:
    task: the corresponding graph task.
    graphs: the list of graphs to use for the task.
    algorithms: the algorithm used to generate the graphs.
    text_encoders: the encoders to use in the tasks.
    cot: whether to apply cot or not.
    random_seed: the random seed to use in the process.
    split: whether we are creating a train or test split.
  """
  random.seed(random_seed)
  zero_shot_examples = utils.create_zero_shot_task(
      task, graphs, algorithms, text_encoders, cot=cot,prompt1=prompt1
  )
  if cot and not prompt1:
    file_name = task.name + '_zero_cot_'
  elif not cot and not prompt1:
    file_name = task.name + '_zero_shot_'
  elif not cot and prompt1:
    file_name = task.name + '_zero_shot_'
  else: 
    file_name = task.name + '_zero_cot_'
  file_name += (prompt1 + split + '.pkl')
  os.makedirs(os.path.join(_TASK_DIR.value, prompt1),exist_ok=True)
  with open(os.path.join(_TASK_DIR.value, prompt1,file_name), 'wb') as f:
    pickle.dump(zero_shot_examples, f)
  zero_shot_examples = [{k:v for k,v in i.items() if k!='graph'} for i in zero_shot_examples]
  df = pd.DataFrame(zero_shot_examples)
  os.makedirs(os.path.join(_TASK_DIR.value, prompt1, "csv"),exist_ok=True)
  df.to_csv(os.path.join(_TASK_DIR.value,prompt1,"csv",file_name.split('.')[0]+'.csv'),
          columns=zero_shot_examples[0].keys(),
          header=zero_shot_examples[0].keys(),
          )

def few_shot(
    task,
    graphs,
    few_shot_graphs,
    algorithms,
    text_encoders,
    cot,
    bag,
    random_seed,
    prompt1='',
    one_shot = False,
):
  """Creating few-shot, cot, or cot-bag examples for the given task.

  Args:
    task: the corresponding graph task.
    graphs: the list of graphs to use for the task.
    few_shot_graphs: the list of graphs to generate few shot examples for.
    algorithms: the algorithm used to generate the graphs.
    text_encoders: the encoders to use in the tasks.
    cot: whether to apply cot or not.
    bag: whether to apply build-a-graph method or not.
    random_seed: the random seed to use in the process.
  """
  random.seed(random_seed)
  few_shot_examples = utils.create_few_shot_task(
      task,
      graphs,
      algorithms,
      few_shot_graphs,
      text_encoders,
      cot=cot,
      bag=bag,
      random_seed=random_seed,
      prompt1=prompt1,
      one_shot = one_shot,
  )
  file_name = task.name
  if not one_shot:
    if cot and bag:
      file_name += f'_cot_bag_{prompt1}test.pkl'
    elif cot:
      file_name += f'_cot_{prompt1}test.pkl'
    elif bag:
      file_name += f'_bag_{prompt1}test.pkl'
    else:
      file_name += f'_few_shot_{prompt1}test.pkl'
  else:
    file_name += f'_one_shot_{prompt1}test.pkl'
  
  os.makedirs(os.path.join(_TASK_DIR.value, prompt1),exist_ok=True)

  with open(os.path.join(_TASK_DIR.value, prompt1,file_name), 'wb') as f:
    pickle.dump(few_shot_examples, f)

  few_shot_examples = [{k:v for k,v in i.items() if k!='graph'} for i in few_shot_examples]
  df = pd.DataFrame(few_shot_examples)
  os.makedirs(os.path.join(_TASK_DIR.value, "csv",prompt1),exist_ok=True)
  df.to_csv(os.path.join(_TASK_DIR.value, "csv",prompt1,file_name.split('.')[0]+'.csv'),
          columns = few_shot_examples[0].keys(),
          header = few_shot_examples[0].keys(),
          )

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _ALGORITHM.value == 'all':
    algorithms = ['er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path']
  else:
    algorithms = [_ALGORITHM.value]
  
  os.makedirs(_TASK_DIR.value,exist_ok=True)
  text_encoders = [
      # low-order 
      "N-Pair",
      "LO-Inc",
      "Adj-Mat",
      # high-order 
      "N-Set",
      "HO-Inc",
      "Inc-Mat",
      "HO-Neigh",
  ]

  # Loading the graphs.
  graphs = []
  generator_algorithms = []
  for algorithm in algorithms:
    loaded_graphs = utils.load_hyper_graphs(
        _GRAPHS_DIR.value,
        algorithm,
        'test',
    )
    graphs += loaded_graphs
    generator_algorithms += [algorithm] * len(loaded_graphs)
  # Defining a task on the graphs
  task = TASK_CLASS[_TASK.value]()
  # NOTE : zero_shot
  zero_shot(
      task,
      graphs,
      generator_algorithms,
      text_encoders,
      cot=False,
      random_seed=_RANDOM_SEED.value,
      split='test',
  )
  # NOTE : zero hyper-cot
  zero_shot(
      task,
      graphs,
      generator_algorithms,
      text_encoders,
      cot=True,
      random_seed=_RANDOM_SEED.value,
      split='test',
  )

  # Loading few-shot graphs.
  few_shot_graphs = []
  for algorithm in algorithms:
    few_shot_graphs += utils.load_hyper_graphs(
        _GRAPHS_DIR.value,
        algorithm,
        'train',
    )
  # NOTE: few shot 
  few_shot(
      task,
      graphs,
      few_shot_graphs,
      generator_algorithms,
      text_encoders,
      cot=False,
      bag=False,
      random_seed=_RANDOM_SEED.value,
  )
  # NOTE: cot
  few_shot(
      task,
      graphs,
      few_shot_graphs,
      generator_algorithms,
      text_encoders,
      cot=True,
      bag=False,
      random_seed=_RANDOM_SEED.value,
  )
  # NOTE: cot hyper-bag
  few_shot(
      task,
      graphs,
      few_shot_graphs,
      generator_algorithms,
      text_encoders,
      cot=True,
      bag=True,
      random_seed=_RANDOM_SEED.value,
  )



if __name__ == '__main__':
  app.run(main)
