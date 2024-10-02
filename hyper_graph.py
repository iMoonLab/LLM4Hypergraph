
import heapq
import dhg 
import itertools
import random
import copy
import numpy as np 
_NUMBER_OF_NODES_RANGE = {
    "small": np.arange(5, 10),
    "medium": np.arange(10, 15),
    "large": np.arange(15, 20),
}
import copy

class HyperGraph(dhg.Hypergraph):
    def __init__(self,vertices,edges) -> None:
        super().__init__(len(vertices),edges)
        self.vertices = vertices
        self.hyperedges = edges
        self.data = {'vertex':vertices,'hypedges':edges}
        pass

    def __getitem__(self, key):
        return self.data[key]
    
    def sample_graph(self):
        """
        sample graphs from real dataset.
        """
        graph_sizes = random.choices(
            list(_NUMBER_OF_NODES_RANGE.keys())
        )
        number_of_vertices = random.choice(_NUMBER_OF_NODES_RANGE[graph_sizes[0]])
        number_of_vertices = min(number_of_vertices,len(self.vertices))
        vertex_list = set()
        while len(vertex_list) < number_of_vertices:
            center = random.choice(self.vertices)
            queue = []
            queue.append(center)
            while len(queue) != 0  and len(vertex_list) < number_of_vertices: 
                neibor_list = set()
                while len(queue) != 0:
                    n = queue.pop(0)
                    tmp = self.neighbor(n)
                    resume_num = max(1,int(random.random() * number_of_vertices*0.5 )) 
                    if len(tmp) >  int(resume_num):
                        tmp = random.sample(tmp,k=int(resume_num))
                    neibor_list.update(set(tmp))
                
                if len(neibor_list) >  int(1/2 *number_of_vertices):
                    neibor_list = random.sample(neibor_list,k=int(1/2 *number_of_vertices))   
                vertex_list.update(set(neibor_list))
                queue.extend(list(set(neibor_list)-vertex_list))
        vertex_list = list(vertex_list)
        vertex_list = vertex_list[:number_of_vertices]
        vertex_list = sorted(vertex_list)
        reorder_dict = {}
        for i,n in  enumerate(vertex_list):
            reorder_dict[n] = i
        edge_set = set()
        edges = copy.deepcopy(self.hyperedges)
        for e in edges:
            com_e = set(e).intersection(set(vertex_list))
            if len(com_e) > 1:
                com_e = [reorder_dict[i] for i in com_e]
                com_e = tuple(sorted(com_e))
                if com_e not in edge_set:
                    edge_set.add(com_e)
        edge_list = sorted(list(edge_set),key= lambda x: x[0])
        if len(edge_list) == 0:
            return self.sample_graph()
        return HyperGraph(range(len(vertex_list)),edge_list)
        
        

    def neighbor(self,source):
        """
        return neighborhood 
        """
        vertex_set = set()
        for e in self.hyperedges:
            if source in e:
                vertex_set.update(set(e))
        if source in vertex_set:
            vertex_set.remove(source)
        return list(vertex_set)



    def shuffleNode(self):
        """
        Randomly shuffle the vertices of the hypergraph for hypergraph isomorphism problems
        """
        verts = copy.deepcopy(self.v)
        random.shuffle(verts)
        hyperedges = []
        for edge in self.e[0]:
            tmp = []
            for n in edge:
                tmp.append(verts[n])
            sorted(tmp,key=int)
            hyperedges.append(tmp)
        return HyperGraph(self.v,hyperedges)

    def keys(self):
        return self.data.keys()
    
    def has_path(self,source,target):
        """
        Whether there is a path between two vertices of the hypergraph, used for reachability tasks
        """
        path = self.short_path(source,target)
        if path is not None:
            return True 
        return False 

    def clique_expanation(self):
        """
        Clique expansion of the hypergraph (preserving edge information, vertex a and vertex b are connected through hyperedges), used for hypergraph textualization clique_low_order_inc
        """
        hyperedges = self.e[0]
        edges = []
        for i,e in enumerate(hyperedges):
            for low_e in list(itertools.combinations(e, 2)):
                low_e = sorted(low_e)
                low_e.append(i)
                edges.append(low_e)
        edges = list(edges)
        self.clique_v = len(self.v)
        self.clique_e = edges
        return len(self.v) , edges
    

    def clique_expanation_low(self):
        """
        Low-order clique extension of the hypergraph (no edge information is preserved, vertex a is connected to vertex b), used for hypergraph textualization clique_adj and clique_inc
        """
        hyperedges = self.e[0]
        edges = set()
        for i,e in enumerate(hyperedges):
            for low_e in list(itertools.combinations(e, 2)):
                low_e = sorted(low_e)
                if tuple(sorted(low_e)) not in edges:
                    edges.add(tuple(sorted(low_e)))
        edges = list(edges)
        self.clique_v = len(self.v)
        self.clique_e = edges
        return len(self.v) , edges

        
    def short_path(self,source,target):
        """
        Shortest path algorithm for hypergraphs
        return: shortest path
        """
        queue = [(0,source)]
        distances = {vertex: float('inf') for vertex in self.vertices}
        previous_vertices = {vertex: None for vertex in self.vertices}

        distances[source] = 0
        while queue:
            current_distance, current_vertex = heapq.heappop(queue)
            if current_vertex == target:
                path = []
                current_vertex = (current_vertex,-1)
                while current_vertex is not None:
                    path.append(current_vertex)
                    current_vertex = previous_vertices[current_vertex[0]]
                return path[::-1]
            if current_distance > distances[current_vertex]:
                continue
            connected_edges = self.edges(current_vertex)
            connected_edges_list = [self.hyperedges[i] for i in connected_edges]
            for i,edges in enumerate(connected_edges_list):
                for vertex in edges:
                    distance = current_distance + 1
                    if distance < distances[vertex]:
                        distances [vertex] = distance
                        previous_vertices[vertex] = (current_vertex,connected_edges[i])
                        heapq.heappush(queue, (distance, vertex))
        return None

    def clique_neighbor(self,source_vertex):
        "Find the neighbor points corresponding to the source vertex after the higher-order clique expansion"
        if not hasattr(self,'clique_e'):
            self.clique_expanation()
        tmp = []
        for i,edge in enumerate(self.clique_e):
            assert len(edge) == 3
            if source_vertex in edge:
                if source_vertex==edge[0]:
                    tmp.append((edge[1],edge[2]))
                elif source_vertex == edge[1]:
                    tmp.append((edge[0],edge[2]))
        try:
            tmp.remove(source_vertex)
        except:
            pass
        return tmp
    

    def clique_neighbor_low(self,source_vertex):
        "Find the neighbor points corresponding to the source vertex after the low-order clique expansion"
        if not hasattr(self,'clique_e'):
            self.clique_expanation_low()
        tmp = []
        for i,edge in enumerate(self.clique_e):
            assert len(edge) == 2
            if source_vertex in edge:
                if source_vertex==edge[0]:
                    tmp.append(edge[1])
                elif source_vertex == edge[1]:
                    tmp.append(edge[0])
        tmp =  sorted(list(set(tmp)))
        try:
            tmp.remove(source_vertex)
        except:
            pass
        return tmp

    def edges(self,vertex):
        """
        return the hyperedge indices contain the vertex 
        """
        tmp = []
        for i,edge in enumerate(self.hyperedges):
            if vertex in edge:
                tmp.append(i)
        return tmp 
    
    
if __name__ == "__main__":
    vertices = [1,2,3,4,5]
    edges = [[1,2],[2,3,4],[4,5]]
    hypergraph = HyperGraph(vertices,edges)
    hypergraph.has_path(1,5)