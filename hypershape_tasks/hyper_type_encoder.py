import json
import os

import hypergraph_text_encoder
from hypershape_tasks.read_hypergraph import get_hypergraphs


def hyper_pyramid_encoder(hypergraph,name_dict):
    share_set = get_share_vertex(hypergraph)
    share_string = ''
    k = 1
    for share_dict in share_set:
        if k == 7:
            share_string = share_string[:-2] + ' and ' + get_share_string(hypergraph, name_dict, share_dict)
            break
        k += 1
        # print(share_dict)
        share_string += get_share_string(hypergraph,name_dict,share_dict)
    # print(share_string)
    explanation = (
        'Hyper_Pyramid: Each hyperedges contains only three vertices, '
        'so the visual shape of each hyperedge is like a triangle. '
        'Because the hyperedges share only one vertex between them in pairs, such as: '
        + share_string +
        'the triangles represented by hyperedges are arranged point-to-point. '
        'In summary, the visualized shape of the hypergraph is like a pyramid stacked by multiple triangles.\n'
    )
    return explanation


def hyper_checked_table_encoder(hypergraph,name_dict):
    share_set = get_share_vertex(hypergraph, 2)
    share_string = ''
    k = 1
    for share_dict in share_set:
        if k == 7:
            share_string = share_string[:-2] + ' and ' + get_share_string(hypergraph, name_dict, share_dict)
            break
        k += 1
        share_string += get_share_string(hypergraph, name_dict, share_dict)
    explanation = (
        'Hyper Checked Table: Each the hyperedges contains only four vertices, '
        'so the visual shape of each hyperedge is like a square. '
        'Because the hyperedges share two vertices between them in pairs, such as: '
        + share_string +
        'the squares represented by the hyperedges are arranged edge to edge. '
        'In summary, the visualized shape of the hypergraph is like a checked table with multiple squares stacked and no gaps.\n'
    )
    return explanation



def hyper_wheel_encoder(hypergraph,name_dict):
    share_vertices_set = []
    for vertice in hypergraph.e[0][0]:
        if vertice in hypergraph.e[0][1] and vertice in hypergraph.e[0][2]:
            share_vertices_set.append(vertice)
    share_vertices_string = ''
    for vertice in share_vertices_set:
        share_vertices_string += name_dict[int(vertice)] + ','
    share_vertices_string = '(' + share_vertices_string[:-1] + ')'
    share_set = get_share_vertex(hypergraph)
    share_string = ''
    k = 1
    for share_dict in share_set:
        if k == 7:
            share_string = share_string[:-2] + ' and ' + get_share_string(hypergraph, name_dict, share_dict)
            break
        k += 1
        share_string += get_share_string(hypergraph, name_dict, share_dict)
    explanation = (
        'Hyper Wheel: All the hyperedges share these vertices '
        + share_vertices_string + ', '
        'so these vertices are like the pivot of a wheel. '
        '''What's more, because the hyperedges share some certain vertices between them in pairs, such as: '''
        + share_string +
        'these hyperedges are connected together in pairs by rotation around the wheel pivot. '
        'In summary, the visualized shape of the hypergraph is like a wheel with some vertices as its pivot and all hyperedges as its hubs.\n'
    )
    return explanation

def hyper_cycle_encoder(hypergraph,name_dict):
    share_set = get_share_vertex(hypergraph)
    share_string = ''
    for share_dict in share_set:
        share_string += get_share_string(hypergraph, name_dict, share_dict)
    explanation = (
        'Hyper Cycle: All the hyperedges share no vertices, they just share partial vertices in pairs, '
        + share_string +
        'a hyperedge has only two adjacent hyperedges, so the visualized shape of the hypergraph is like a cycle which is end-to-end.\n'
    )
    return explanation



def get_share_vertex(hypergraph, count=0):
    share_set = []
    if count == 0:
        for i in range(len(hypergraph.e[0])-1):
            for j in range(i+1,len(hypergraph.e[0])):
                share_vertices = set(hypergraph.e[0][i]).intersection(set(hypergraph.e[0][j]))
                if len(share_vertices) > 0:
                    share_set.append({'hyperE_1':i,'hyperE_2':j,'share_vertices':list(share_vertices)})
    else:
        for i in range(len(hypergraph.e[0])-1):
            for j in range(i+1,len(hypergraph.e[0])):
                share_vertices = set(hypergraph.e[0][i]).intersection(set(hypergraph.e[0][j]))
                if len(share_vertices) == count:
                    share_set.append({'hyperE_1':i,'hyperE_2':j,'share_vertices':list(share_vertices)})
    return share_set

def get_share_string(hypergraph,name_dict,share_dict):
    share_string='hyperedge ('
    for vertex in hypergraph.e[0][share_dict['hyperE_1']]:
        share_string += name_dict[int(vertex)] + ','
    share_string = share_string[:-1] + ') is connected to hyperedge('
    for vertex in hypergraph.e[0][share_dict['hyperE_2']]:
        share_string += name_dict[int(vertex)] + ','
    if len(share_dict['share_vertices']) == 1:
        share_string = share_string[:-1] + ') by vertex ('
    else:
        share_string = share_string[:-1] + ') by vertices ('
    for vertex in share_dict['share_vertices']:
        share_string += name_dict[int(vertex)] + ','
    share_string = share_string[:-1] + '), '
    return share_string

if __name__ == '__main__':
    hypergraphs, answers = get_hypergraphs("./RHG-data/RHG_10.txt")
    text_encoders = [
        "N-Pair",
        "LO-Inc",
        "Adj-Mat",
        "N-Set",
        "HO-Inc",
        "Inc-Mat",
        "HO-Neigh",
    ]
    name_dict = hypergraph_text_encoder.NODE_ENCODER_DICT[text_encoders[0]]
    string_dict0 = {}
    string_dict1 = {}
    string_dict2 = {}
    string_dict3 = {}
    string_dict4 = {}
    string_dict5 = {}
    string_dict6 = {}
    string_dict7 = {}
    string_dict8 = {}
    string_dict9 = {}

    for ind,hypergraph in enumerate(hypergraphs):
        if '2' in answers[ind]:
            string_dict2.setdefault(str(ind), hyper_pyramid_encoder(hypergraph, name_dict))
        elif '4' in answers[ind]:
            string_dict4.setdefault(str(ind), hyper_checked_table_encoder(hypergraph, name_dict))
        elif '6' in answers[ind]:
            string_dict6.setdefault(str(ind), hyper_wheel_encoder(hypergraph, name_dict))

    file_name = 'hyper_cycle_encoder.txt'
    with open(os.path.join("./hypergraphqa/hypershape_tasks", file_name), 'w') as f:
        f.write('Hyper_Cycle\n')
        for key,value in string_dict7.items():
            f.write(f"key: {key}, value: {value}\n")
        f.close()