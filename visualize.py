from itertools import product
from sys import stderr

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from A import (Circumstances, Day, Demand, Demands, Graph, Shelters, io_1,
               range_1idx)


def submission() -> None:
    print(0)
    print(0)
    print("submit", 0, 0)



if __name__ == "__main__":
    circumstances = Circumstances(io_1("budget"), io_1("temporal"))
    graph = Graph(io_1("graph")); Demand.graph = graph

    demands = Demands(io_1("demand"), circumstances)
    for d, i in product(circumstances.days(), range_1idx(demands.N)):
        demand = Demand(d, i, io_1("demand", d, i))
        demands[d, i] = demand
    else:
        del demand
    
    shelters = Shelters(io_1("shelter"), circumstances, graph)


    nx.draw(graph._value, pos={v: v.pos._value for v in graph.vertices}, node_size=2)

    vertexpoints = np.array([v.pos._value for v in graph.vertices])

    demandpoints = np.array([d.vertex.pos._value for d in demands[Day(1)]])
    shelterpoints = np.array([s.vertex.pos._value for s in shelters._value])

    plt.scatter(*demandpoints.T, c="green", zorder=2)
    plt.scatter(*shelterpoints.T, c="red", zorder=2)

    plt.show()

    submission()
