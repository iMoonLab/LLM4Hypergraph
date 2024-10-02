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

"""Creates a dictionary mapping integers to vertex names."""

import random

_RANDOM_SEED = 1234
random.seed(_RANDOM_SEED)

_INTEGER_NAMES = [
    str(i) for i in range(200)
]
def create_name_dict(name, nvertices = 20):
  """The runner function to map integers to vertex names.

  Args:
    name: name of the approach for mapping.
    nvertices: optionally provide nvertices in the graph to be encoded.

  Returns:
    A dictionary from integers to strings.
  """
  if name == "integer":
    names_list = _INTEGER_NAMES
  elif name == "random_integer":
    names_list = []
    for _ in range(nvertices):
      names_list.append(str(random.randint(0, 1000000)))
  else:
    raise ValueError(f"Unknown approach: {name}")
  name_dict = {}
  for ind, value in enumerate(names_list):
    name_dict[ind] = value
  return name_dict
