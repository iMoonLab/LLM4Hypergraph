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
import hypergraph_task_utils as utils
import pandas as pd

import hypergraph_task
from read_hypergraph import get_hypergraphs

_TASK = flags.DEFINE_enum(
    'task',
    None,
    [
        'StructureClassification',
    ],
    'The task to generate datapoints.',
    required=True,
)
_ALGORITHM = flags.DEFINE_enum(
    'algorithm',
    None,
    ['hypergraph'],
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
    'StructureClassification': hypergraph_task.StructureClassification,
}


def zero_shot(
        task,
        graphs,
        algorithms,
        text_encoders,
        answers,
        cot,
        random_seed,
        split,
        prompt1=''
):
    """Creating zero-shot or zero-cot examples for the given task.

    Args:
      task: the corresponding graph task.
      graphs: the list of graphs to use for the task.
      algorithms: the algorithm used to generate the graphs.
      text_encoders: the encoders to use in the tasks.
      answers: the category to which a hypergraph structure belongs.
      cot: whether to apply cot or not.
      random_seed: the random seed to use in the process.
      split: whether we are creating a train or test split.
    """
    random.seed(random_seed)
    zero_shot_examples = utils.create_zero_shot_task(
        task, graphs, algorithms, text_encoders, answers, cot=cot, prompt1=prompt1
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
    os.makedirs(os.path.join(_TASK_DIR.value, prompt1), exist_ok=True)
    with open(os.path.join(_TASK_DIR.value, prompt1, file_name), 'wb') as f:
        pickle.dump(zero_shot_examples, f)
    zero_shot_examples = [{k: v for k, v in i.items() if k != 'graph'} for i in zero_shot_examples]
    df = pd.DataFrame(zero_shot_examples)
    os.makedirs(os.path.join(_TASK_DIR.value, prompt1, "csv"), exist_ok=True)
    df.to_csv(os.path.join(_TASK_DIR.value, prompt1, "csv", file_name.split('.')[0] + '.csv'),
              columns=zero_shot_examples[0].keys(),
              header=zero_shot_examples[0].keys(),
              )

def few_shot(
        task,
        graphs,
        few_shot_graphs,
        algorithms,
        text_encoders,
        answers,
        answers_example,
        cot,
        bag,
        random_seed,
):
    """Creating few-shot, cot, or cot-bag examples for the given task.

    Args:
      task: the corresponding graph task.
      graphs: the list of graphs to use for the task.
      few_shot_graphs: the list of graphs to generate few shot examples for.
      algorithms: the algorithm used to generate the graphs.
      text_encoders: the encoders to use in the tasks.
      answers: the category to which a hypergraph structure belongs.
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
        answers,
        answers_example,
        cot=cot,
        bag=bag,
        random_seed=random_seed,
    )
    file_name = task.name

    if cot and bag:
        file_name += '_cot_bag_test.pkl'
    elif cot:
        file_name += '_cot_test.pkl'
    else:
        file_name += '_few_shot_test.pkl'
    os.makedirs(os.path.join(_TASK_DIR.value), exist_ok=True)

    with open(os.path.join(_TASK_DIR.value, file_name), 'wb') as f:
        pickle.dump(few_shot_examples, f)

    few_shot_examples = [{k: v for k, v in i.items() if k != 'graph'} for i in few_shot_examples]
    df = pd.DataFrame(few_shot_examples)
    os.makedirs(os.path.join(_TASK_DIR.value, "csv"), exist_ok=True)
    df.to_csv(os.path.join(_TASK_DIR.value, "csv", file_name.split('.')[0] + '.csv'),
              columns=few_shot_examples[0].keys(),
              header=few_shot_examples[0].keys(),
              )

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    
    algorithms = [_ALGORITHM.value]

    os.makedirs(_TASK_DIR.value, exist_ok=True)
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
    generator_algorithms = []
    # max_edge25,max_vertex100,almost 100
    hypergraphs, answers = get_hypergraphs("./RHG-data/RHG_10.txt",
                                           type=[2, 4, 6])

    for algorithm in algorithms:
        generator_algorithms += [algorithm] * len(hypergraphs)

    # Defining a task on the graphs
    task = TASK_CLASS[_TASK.value]()
    # Loading few-shot graphs.
    # max_edge15,max_vertex50,almost 50
    hypergraphs_example, answers_example = get_hypergraphs("./hypergraphqa/RHG-data/RHG_10.txt",
                                                           max_v=15, max_n=50, )

    # NOTE : zero_shot
    zero_shot(
        task,
        hypergraphs,
        generator_algorithms,
        text_encoders,
        answers,
        cot=False,
        random_seed=1234,
        split='test',
    )
    # NOTE : zero hyper-cot
    zero_shot(
        task,
        hypergraphs,
        generator_algorithms,
        text_encoders,
        answers,
        cot=True,
        random_seed=1234,
        split='test',
    )
    zero_shot(
        task,
        hypergraphs,
        generator_algorithms,
        text_encoders,
        answers,
        cot=False,
        random_seed=1234,
        split='test',
        prompt1='v1'  
    )

    # NOTE: few shot
    few_shot(
        task,
        hypergraphs,
        hypergraphs_example,
        generator_algorithms,
        text_encoders,
        answers,
        answers_example,
        cot=False,
        bag=False,
        random_seed=1234,
    )
    # NOTE: cot
    few_shot(
        task,
        hypergraphs,
        hypergraphs_example,
        generator_algorithms,
        text_encoders,
        answers,
        answers_example,
        cot=True,
        bag=False,
        random_seed=1234,
    )
    # NOTE: cot hyper-bag
    few_shot(
        task,
        hypergraphs,
        hypergraphs_example,
        generator_algorithms,
        text_encoders,
        answers,
        answers_example,
        cot=True,
        bag=True,
        random_seed=1234,
    )


if __name__ == '__main__':
    app.run(main)