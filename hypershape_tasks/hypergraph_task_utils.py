import random
import re


def create_example_feature(
    key,
    question,
    answer,
    algorithm,
    encoding_method,
    vertices,
    nedges,
    graph = None
):
  """Create a tensorflow example from a datapoint."""
  key_feature = key
  question_feature = question

  answer_feature = answer
  algorithm_feature = algorithm
  encoding_method_feature = encoding_method
  nvertices_feature = vertices
  nedges_feature = nedges
  # json形式
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

def create_zero_shot_task(
        task,
        graphs,
        generator_algorithms,
        text_encoders,
        answers,
        cot=False,
        prompt1=''
):
    """Create a recordio file with zero-shot examples for the task."""
    examples = []
    for encoding_method in text_encoders:
        examples_dict = task.prepare_examples_dict(
            graphs, generator_algorithms, encoding_method, answers
        )
        # print(prompt1)
        if cot:
            for key in examples_dict.keys():
                examples_dict[key]['question'] += "Let's think step by step. "
        if prompt1 == 'v1':
            for key in examples_dict.keys():
                examples_dict[key][
                    'question'] += "Let's think step by step. Make sure the data is calculated and recorded accurately at each step."
        elif prompt1 == 'v2':
            for key in examples_dict.keys():
                examples_dict[key][
                    'question'] += "Let's analyze the connectivity by considering hyperedges linked to vertices and vertices linked through hyperedges. "
        elif prompt1 == 'v3':
            for key in examples_dict.keys():
                examples_dict[key][
                    'question'] += "Let's think hyperedges connected by vertices then vertices connected by hyperedges."
        examples += prepare_examples(examples_dict, encoding_method)
    return examples

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

def prepare_few_shots(
    task,
    graphs,
    text_encoders,
    answers,
    cot,
):
  """Create a dict of few-shot examples with their cot for the task."""
  few_shots_examples_dict = {}
  for encoding_method in text_encoders:
    if encoding_method not in few_shots_examples_dict:
      few_shots_examples_dict[(encoding_method)] = []
    for ind,graph in enumerate(graphs):
      few_shots_examples_dict[(encoding_method)].append(
        task.create_few_shot_example(graph,encoding_method,answers[ind][0],cot)
      )
  return few_shots_examples_dict


def choose_few_shot_examples(
    few_shots_dict,
    encoding_method,
    k = 3,
):
  pattern = r'Ans:\s*\[(\d+),?\]'
  """Choose few shot examples for each algorithm."""
  few_shots_str = ''
  example_list = few_shots_dict[encoding_method]
  for i in [2,4,6]: # the type to choose
    example_type_list = []
    for example in example_list:
        if int(re.search(pattern, example).group(1)) == i:
            example_type_list.append(example)
    few_shots_str += 'Example: ' + random.choice(example_type_list) + '\n'
  return few_shots_str

def create_few_shot_task(
    task,
    graphs,
    generator_algorithms,
    few_shots_graphs,
    text_encoders,
    answers,
    answers_example,
    cot,
    bag,
    random_seed,
):
  """Create a recordio file with few-shot examples for the task."""
  number_of_tokens = {}
  examples = []
  print('prepare few shot task', 'cot', cot, 'bag', bag)
  few_shots_examples_dict = prepare_few_shots(
      task,
      few_shots_graphs,
      text_encoders,
      answers_example,
      cot,
  )
  for encoding_method in text_encoders:
    random.seed(random_seed)
    examples_dict = task.prepare_examples_dict(
        graphs, generator_algorithms, encoding_method, answers
    )
    for key in examples_dict.keys():
      few_shots_examples = choose_few_shot_examples(
          few_shots_examples_dict,
          encoding_method,
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
