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

import hypergraph_text_encoder

from hyper_graph import HyperGraph
from hypershape_tasks import hyper_type_encoder


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

TYPE_ENCODER={
    '2': hyper_type_encoder.hyper_pyramid_encoder,
    '4': hyper_type_encoder.hyper_checked_table_encoder,
    '6': hyper_type_encoder.hyper_wheel_encoder,
}

class StructureClassification(GraphTask):
    """The graph task to predict the shape of the hypergraph"""

    def __init__(self):
        super().__init__()
        self.name = 'shape_prediction'

    def prepare_examples_dict(
            self,
            graphs,
            generator_algorithms,
            encoding_method,
            answers,
    ):
        examples_dict = {}
        name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]

        for ind, graph in enumerate(graphs):
            question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
            task_description = (
                'Q: What is the shape of the visualization of the hypergraph like?'
                ' Please directly give the answer number corresponding to the 3 hypergraph visualization shapes as shown below.'
                ' Answer [2] corresponds to the Hyper Pyramid, which represents the visualization of the hypergraph as a Pyramid.'
                ' Answer [4] corresponds to the Hyper Checked Table, which represents the visualization of the hypergraph as a Checked Table.'
                ' Answer [6] corresponds to the Hyper Wheel, which represents the visualization of the hypergraph as a Wheel.'
                '''\nList the answer after "Ans:" in the format like [2]. There is only one unique sure answer to this question, so don't give a list of possible answers.\nA: '''
            )
            question += task_description
            answer = answers[ind]
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

    def create_few_shot_example(
            self, graph, encoding_method, answer, cot
    ):
        name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[encoding_method]
        question = hypergraph_text_encoder.encode_graph(graph, encoding_method)
        question += (
            'Q: What is the shape of the visualization of the hypergraph like?'
            ' Please directly give the answer number corresponding to the 3 hypergraph visualization shapes as shown below.'
            ' Answer [2] corresponds to the Hyper Pyramid, which represents the visualization of the hypergraph as a Pyramid.'
            ' Answer [4] corresponds to the Hyper Checked Table, which represents the visualization of the hypergraph as a Checked Table.'
            ' Answer [6] corresponds to the Hyper Wheel, which represents the visualization of the hypergraph as a Wheel.'
            ''' List the answers after "Ans:" in the format like [2]. There is only one unique sure answer to this question, so don't give a list of possible answers.\nA: '''
        )

        explanation = TYPE_ENCODER[str(answer)](graph,name_dict)
        answer = 'Ans:[' + answer + ']. '
        answer += explanation

        return question + answer