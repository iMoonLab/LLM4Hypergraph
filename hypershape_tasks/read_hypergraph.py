import dhg
import matplotlib.pyplot as plt

from hyper_graph import HyperGraph


def get_hypergraphs(file_path,type=None,max_v=25,max_n=100):
    hypergraphs = []
    answers = []
    max_vertex = 0
    max_edge = 0
    with open(file_path, 'r') as file:
        start_num = 1
        lines = file.readlines()
        while start_num < len(lines):
            elements = lines[start_num].split(' ')
            num_hyper_v = elements[0]
            num_hyper_e = elements[1]
            if int(num_hyper_v) > max_vertex:
                max_vertex = int(num_hyper_v)
            if int(num_hyper_e) > max_edge:
                max_edge = int(num_hyper_e)
            end_num = start_num + 2 + int(num_hyper_e)
            if int(num_hyper_v) < max_v and int(num_hyper_e) < max_n:
                if type is None:
                    answers.append(elements[2].rstrip() + ',')
                else:
                    if int(elements[2]) in type:
                        answers.append(elements[2].rstrip() + ',')
                hyperedges = []
                for line in lines[start_num+2:end_num]:
                    hyperedge = [int(i) for i in line.split(' ')]
                    hyperedges.append(hyperedge)
                hypervertices = [i for i in range(int(num_hyper_v))]
                hypergraph = HyperGraph(hypervertices, hyperedges)
                # vis
                # hg = dhg.Hypergraph(int(num_hyper_v), hyperedges)
                # hg.draw()
                # plt.show()
                # print("answer：",elements[2].rstrip())
                if type is None:
                    hypergraphs.append(hypergraph)
                else:
                    # print(elements[2])
                    if int(elements[2]) in type:
                        hypergraphs.append(hypergraph)
            start_num = end_num
    print("answers:",answers)
    print("max_edge:",max_edge)
    print("max_vertex:", max_vertex)
    return hypergraphs,answers


if __name__ == "__main__":
  hypergraphs,_ = get_hypergraphs(
      file_path = "/home2/houxingliang/hypergraphqa/RHG-data/RHG_10.txt",
      type=6,
  )
  print(len(hypergraphs))
  print("hypergraphs:",hypergraphs)
