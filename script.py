import networkx as nx
import random as rand
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt

def build_graph(n = 10000, c = 4):
    p = c/(n-1)
    G = nx.Graph()
    nodes = [i for i in range(n)]
    G.add_nodes_from(nodes, cluster = 0 )
    for i in range(len(nodes)):
        probs = [rand.random() for node in nodes]
        for j in range(i, len(nodes)):
            if probs[j] < p and i != j:
                G.add_edge(nodes[i], nodes[j])
    return G

def largest_cluster(cluster_list, cluster_to_nodes):
    max_cluster_size = 0
    max_cluster = 0
    for cluster in cluster_list:
        cluster_size = len(cluster_to_nodes[cluster])
        if cluster_size > max_cluster_size:
            max_cluster_size = cluster_size
            max_cluster = cluster
    return max_cluster

def percolate (g):
    cluster_to_nodes = defaultdict(list)
    new_g = nx.Graph(cluster = 0)
    nodes_to_add = rand.sample(list(g.nodes), len(list(g.nodes)))
    edges = list(g.edges)
    num_nodes = len(nodes_to_add)
    cluster_num = 1
    biggest_cluster = [0.0] * num_nodes
    for t in range(num_nodes):
        #Random sample from the nodes
        node = nodes_to_add[t]
        #Add the random node
        new_g.add_node(node, cluster = 0)
        cluster_list = set()
        for edge in g.edges(node):
            # Get the other end
            if edge[0] == node:
                other_node = edge[1]
            else:
                other_node = edge[0]
            if new_g.has_node(other_node):
                new_g.add_edge(node, other_node)
                #Get list of unique clusters of the nodes we connect to
                cluster_list.add(new_g.nodes[other_node]["cluster"])

        # Join to the biggest cluster
        if len(cluster_list) != 0:
            cluster_to_join = largest_cluster(cluster_list, cluster_to_nodes)
            new_g.nodes[node]["cluster"] = cluster_to_join
            cluster_to_nodes[cluster_to_join].append(node)
        else:
            new_g.nodes[node]["cluster"] = cluster_num
            cluster_to_nodes[cluster_num].append(node)
            cluster_num += 1
        # For each other cluster, merge it to this one
        for cluster in cluster_list:
            connected_nodes = cluster_to_nodes[cluster]
            cluster_to_nodes.pop(cluster)
            for c_node in connected_nodes:
                # Update cluster_to_nodes for the merged node
                cluster_to_nodes[new_g.nodes[node]["cluster"]].append(c_node)
                new_g.nodes[c_node]["cluster"] =  new_g.nodes[node]["cluster"]

        cluster_counts = Counter(nx.get_node_attributes(new_g, "cluster").values())
        biggest_cluster[t] =  cluster_counts.most_common(1)[0][1]
    return biggest_cluster

if __name__ == '__main__':
    g = []
    num_graphs = 10
    num_t = 1000
    biggest_clusters = []
    for i in range(num_graphs):
        g.append(build_graph(n=num_t))
    x_axis = [t for t in range(num_t)]
    results = [percolate(g[i]) for i in range(num_graphs)]
    averages = []
    for t in range(num_t):
        size_for_t = [val[t] for val in results]
        averages.append(sum(size_for_t)/float(len(size_for_t)))
    plt.plot(x_axis, averages)
    plt.show()

