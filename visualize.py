from __future__ import annotations

import matplotlib.pyplot as plt

from A import *


## main func
def main(
    circumstances: Circumstances,
    score_calc   : ScoreCalculator,
    graph        : Graph,
    demands      : Demands,
    radiation    : Radiation,
    assets       : AssetCatalog,
    order_freq   : OrderFrequency,
    shelters     : Shelters,
) -> None:
    nx.draw(graph._value, pos={v: v.pos._value for v in graph.vertices}, node_size=2)

    demandpoints = np.array([d.vertex.pos._value for d in demands[Day(1)].values()])
    shelterpoints = np.array([s.vertex.pos._value for s in shelters._value])

    plt.scatter(*demandpoints.T, c="green", zorder=2)
    plt.scatter(*shelterpoints.T, c="red", zorder=2)

    plt.show()

    io_2(Answer(), command="submit", opt=0)




## processing
if __name__ == "__main__":
    # Input and Output 1
    circumstances = Circumstances(io_1("budget"), io_1("temporal"))
    
    graph = Graph(io_1("graph"))
    Vehicle.graph = Demand.graph = graph
    
    score_calc = ScoreCalculator(io_1("score"), circumstances, graph)
    
    demands = Demands(io_1("demand"), circumstances)
    for d, i in product(circumstances.days(), range_1idx(demands.N)):
        demand = Demand(d, i, io_1("demand", d, i))
        demands[d, i] = demand
    else:
        del demand
    
    radiation = Radiation(circumstances, graph)
    for d, v in product(circumstances.days(), graph.vertices):
        radiation[d, v] = io_1("radiation", d, v.id)
    assets = AssetCatalog(io_1("asset"))
    for i in range_1idx(assets.N_PV):
        PV.add_variety(*io_1("asset", "PV", i))
    for i in range_1idx(assets.N_FE):
        FE.add_variety(*io_1("asset", "FE", i))
    for i in range_1idx(assets.N_RB):
        RB.add_variety(*io_1("asset", "RB", i))
    for i in range_1idx(assets.N_EVC):
        EVC.add_variety(*io_1("asset", "EVC", i))
    for i in range_1idx(assets.N_V):
        l0, l1 = io_1("asset", "vehicle", i)
        Vehicle.add_variety(*l0, *l1)
    
    order_freq = OrderFrequency(circumstances, graph)
    for d in circumstances.days():
        order_freq[d] = io_1("order", d)
    
    shelters = Shelters(io_1("shelter"), circumstances, graph)
    
    io_1("end")


    main(
        circumstances,
        score_calc,
        graph,
        demands,
        radiation,
        assets,
        order_freq,
        shelters,
    )








