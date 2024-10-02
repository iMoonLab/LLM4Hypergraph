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

r"""Random hypergraph generation."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import networkx as nx
from tensorflow.io import gfile
# from hyper_graph import HyperGraph
# from graphqa import graph_generator_utils
import hypergraph_generator_utils
import itertools

def clique_expanation(graph):
    hyperedges = graph.e[0]
    # edges = []
    edges = set()
    for e in hyperedges:
        for low_e in list(itertools.combinations(e, 2)):
            if tuple(sorted(list(low_e))) not in edges:
                edges.add(tuple(sorted(list(low_e))))
    edges = list(edges)
    return len(edges)*2/(len(graph.v)*(len(graph.v)-1))
_ALGORITHM = flags.DEFINE_string(
    "algorithm",
    "hypergraph_high",
    "The graph generating algorithm to use.",
    required=False,
)
_NUMBER_OF_GRAPHS = flags.DEFINE_integer(
    "number_of_graphs",
    500,
    "The number of graphs to generate.",
    required=False,
)
_DIRECTED = flags.DEFINE_bool(
    "directed", False, "Whether to generate directed graphs."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", '/home/yangchengwu/home2/Hyper_2024/hypergraphqa/hypergraphs/Ablation/edge_degree', "The output path to write the graphs.", required=False
)
_SPLIT = flags.DEFINE_string(
    "split", 'test', "The dataset split to generate.", required=False
)
_MIN_SPARSITY = flags.DEFINE_float("min_sparsity", 0.0, "The minimum sparsity.")
_MAX_SPARSITY = flags.DEFINE_float("max_sparsity", 1.0, "The maximum sparsity.")

import dhg 
def write_graphs(graphs, output_dir):
  """Writes graphs to output_dir."""
  if not gfile.exists(output_dir):
    gfile.makedirs(output_dir)
  if isinstance(graphs[0],dict) or isinstance(graphs[0],dhg.structure.Hypergraph):
    for ind, graph in enumerate(graphs):
      hypergraph_generator_utils.write_graph_pkl(graph,os.path.join(output_dir, str(ind) + ".pkl"))
  else:
    for ind, graph in enumerate(graphs):
      nx.write_graphml(
          graph,
          open(
              os.path.join(output_dir, str(ind) + ".graphml"),
              "wb",
          ),
      )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _SPLIT.value == "train":
    random_seed = 9876
  elif _SPLIT.value == "test":
    random_seed = 1234
  elif _SPLIT.value == "valid":
    random_seed = 5432
  else:
    raise NotImplementedError()

  generated_graphs = hypergraph_generator_utils.generate_graphs(
      number_of_graphs=_NUMBER_OF_GRAPHS.value,
      algorithm=_ALGORITHM.value,
      directed=_DIRECTED.value,
      random_seed=random_seed,
      er_min_sparsity=_MIN_SPARSITY.value,
      er_max_sparsity=_MAX_SPARSITY.value,
  )
  # rate = [clique_expanation(g) for g in generated_graphs]
  write_graphs(
      graphs=generated_graphs,
      output_dir=os.path.join(
          _OUTPUT_PATH.value,
          # _ALGORITHM.value,
          _SPLIT.value,
      ),
  )


if __name__ == "__main__":
  app.run(main)
