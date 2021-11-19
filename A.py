from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import (Any, Callable, ClassVar, Iterator, List, Literal,
                    NamedTuple, Tuple, Union, overload)

import networkx as nx
import numpy as np




## self-developed utility classes and functions
class IO:
    @staticmethod
    def str0d() -> str:
        return input()

    @staticmethod
    def str1d(n:int|None=None) -> list[str]:
        if n is None:
            return open(0).readlines()
        return [IO.str0d() for _ in range(n)]


    @staticmethod
    def int0d() -> int:
        return int(input())

    @staticmethod
    def int1d() -> list[int]:
        return list(map(int, input().split()))

    @staticmethod
    def int2d(n:int|None=None) -> list[list[int]]:
        if n is None:
            return [list(map(int, l.split())) for l in open(0).readlines()]
        return [IO.int1d() for _ in range(n)]

    
    @staticmethod
    def float0d() -> float:
        return float(input())

    @staticmethod
    def float1d() -> list[float]:
        return list(map(float, input().split()))

    @staticmethod
    def float2d(n:int|None=None) -> list[list[float]]:
        if n is None:
            return [list(map(float, l.split())) for l in open(0).readlines()]
        return [IO.float1d() for _ in range(n)]


    @staticmethod
    def YNeos(b:bool, s:str='YNeos') -> None:
        print(s[b::2])

    @staticmethod
    def print(*i: Any) -> None:
        print(*i)


    @staticmethod
    def errprint(*i: Any) -> None:
        from sys import stderr
        print(*i, file=stderr)



class InstanceCacheMixIn:
    instances: ClassVar[defaultdict[type, dict[Any, InstanceCacheMixIn]]] = defaultdict(dict)
    def __new__(cls: type[InstanceCacheMixIn], *key: Any) -> InstanceCacheMixIn:
        value = cls.instances[cls].setdefault(
            key,
            super(InstanceCacheMixIn, cls).__new__(cls, *key)
        )
        return value



def range_1idx(length: int) -> range:
    return range(1, length + 1)




## Classes
# Budget, Temporal infomation
class Circumstances:
    def __init__(self, binfo: int, tinfo: list[int]) -> None:
        self.budget = binfo
        (
            self.steps_per_day,
            self.last_order_step,
            self.intervals_per_day,
            self.number_of_days,
            self.accident_days,
        ) = tinfo

    @property
    def steps_per_interval(self) -> int:
        return self.steps_per_day // self.intervals_per_day


    def steps(self) -> Iterator[Step]:
        "yield steps until the DAY finish: not the INTERVAL."
        for i in range_1idx(self.steps_per_day):
            yield Step(i)

    def intervals(self) -> Iterator[Interval]:
        for i in range_1idx(self.intervals_per_day):
            yield Interval(i)

    def days(self) -> Iterator[Day]:
        for i in range_1idx(self.number_of_days):
            yield Day(i)

    def steps_in_a_interval(self) -> Iterator[Step]:
        for i in range_1idx(self.steps_per_interval):
            yield Step(i)


class Day(int, InstanceCacheMixIn): ...
class Interval(int, InstanceCacheMixIn): ...
class Step(int, InstanceCacheMixIn): ...



# score caluculator maker
class ScoreCalculator:
    def __init__(self, info: _ScoreInfo, c: Circumstances, g: Graph) -> None:
        (
            (self.alpha_cost,),
            self.w_day,
            (self.w_trans, self.w_ele, self.w_env, self.w_acc,),
            (self.alpha_trans_fee, self.alpha_trans_penalty,                    ),
            (self.alpha_ele,       self.alpha_ele_FE,        self.alpha_ele_buy,),
            (self.alpha_env_fuel,  self.alpha_env_buy,                          ),
            (self.alpha_acc,                                                    ),
        ) = info
        self.circumstances = c
        self.graph = g


    def __call__(self, ans: Answer, res: Result) -> float:
        days = self.circumstances.days()
        budget = self.circumstances.budget
        score =\
            sum(self.w_day[d - 1] * self._daily_score(d, res) for d in days)\
          + self.w_acc * self._disaster_response_score(res)\
          - self.alpha_cost * max(0, ans.initial_cost - budget)
        return score


    def _daily_score(self, d: Day, res: Result) -> float:
        score =\
            self.w_trans * self._transportation_score(d, res)\
          + self.w_ele   * self._energy_score(d, res)\
          + self.w_env   * self._environmental_score(d, res)
        return score


    def _transportation_score(self, d: Day, res: Result) -> float:
        def dist(o: Order) -> int:
            return self.graph.distance(o.v_pickup, o.v_destination)
        score = sum(
            max(
                0,
                self.alpha_trans_fee * (ti := dist(o))\
              - self.alpha_trans_penalty * (o.time_spent - ti)**2
            )
            for o in res.orders_Tmax(d)
        )
        return score


    def _energy_score(self, day: Day, res: Result) -> float:
        C_valance = sum(
            ev.Chg_current - ev.Chg_init - ev.Delta_move * ev.return_distance()
            for ev in res.evs_Tmax(day)
        ) + sum(
            grid.Chg_current - grid.Chg_init
            for grid in res.grids_Tmax(day)
        )
        score = self.alpha_ele * C_valance\
              - self.alpha_ele_FE * res.total_FE_power_generation(day)\
              - self.alpha_ele_buy * res.total_power_purchase(day)
        return score


    def _environmental_score(self, day: Day, res: Result) -> float:
        score = - self.alpha_env_fuel * res.total_FE_power_generation(day)\
                - self.alpha_env_buy * res.total_power_purchase(day)
        return score


    def _disaster_response_score(self, res: Result) -> float:
        score = self.alpha_acc * res.uninterruptible_days()
        return score



# Graph Classes
class Position:
    def __init__(self, x: int, y: int) -> None:
        self._value = np.array([x, y], dtype=np.int64)

    @staticmethod
    def euclidean(p0: Position, p1: Position) -> float:
        return np.linalg.norm(p0._value - p1._value)


class Graph:
    def __init__(self, veinfo: tuple[list[int], list[list[int]], list[list[int]]]) -> None:
        self.e: int; self.v: int
        vinfo: list[list[int]]; einfo: list[list[int]]
        ((self.v, self.e), vinfo, einfo) = veinfo

        self._value: nx.Graph[Vertex] = nx.Graph()
        self.vertices = vs = [
            Vertex(i, Position(x, y), p, A, l)
            for i, (x, y, p, A, l) in enumerate(vinfo, 1)
        ]
        self.edges = es = [
            Edge(vs[u - 1], vs[v - 1], d)
            for u, v, d in einfo
        ]
        self._value.add_nodes_from(vs)
        self._value.add_weighted_edges_from(es)

        self.scale = max(Position.euclidean(e.u.pos, e.v.pos)/e.weight for e in es)

    def getvertex(self, i: int) -> Vertex:
        return self.vertices[i - 1]

    def get_edge_dist(self, u: Vertex, v: Vertex) -> int:
        return self._value.get_edge_data(u, v)["weight"]

    def distance(self, u: Vertex, v: Vertex) -> int:
        def euclidean(a: Vertex, b: Vertex):
            return self.scale * Position.euclidean(a.pos, b.pos)
        value: int = nx.astar_path_length(self._value, u, v, euclidean)
        return value


class Edge(NamedTuple):
    u: Vertex
    v: Vertex
    weight: int


class Vertex(NamedTuple):
    id: int
    pos: Position
    pop: int
    area: int
    cost_per_area: int

    def __str__(self) -> str:
        return str(self.id)



# Demand Classes
class Demands:
    def __init__(self, N: int, c: Circumstances) -> None:
        self.N = N
        self.circumstances = c
        self._value: list[list[Demand]] = [
            [
                Demand.empty() for _ in range(N)
            ] for _ in range(c.number_of_days)
        ]


    @overload
    def __getitem__(self, key: Day | tuple[Day]) -> list[Demand]: ...

    @overload
    def __getitem__(self, key: tuple[Day, int]) -> Demand: ...

    @overload
    def __getitem__(self, key: tuple[Day, int, Interval]) -> int: ...

    def __getitem__(self, key: Day | tuple[Day] | tuple[Day, int] | tuple[Day, int, Interval]):
        if isinstance(key, int):
            return self._value[key - 1]
        else:
            l = len(key)
            if   l == 1:
                dl = self._value[key[0] - 1]
                return dl
            elif l == 2:
                d = self._value[key[0] - 1][key[1] - 1]
                return d
            elif l == 3:
                d = self._value[key[0] - 1][key[1] - 1]
                if isinstance(key[2], Interval):
                    return d[key[2]]
        raise KeyError

    def __setitem__(self, key: tuple[Day, int], value: Demand) -> None:
        day: Day; n: int
        day, n = key
        self._value[day - 1][n - 1] = value


class Demand:
    graph: ClassVar[Graph]
    _empty: Demand | None = None

    def __init__(self, day: Day, id: int, info: list[list[int]]) -> None:
        self.day = day
        self.id  = id
        (
            (x, self.sigma2),
            self.daily_demands
        ) = info

        self.vertex = self.graph.getvertex(x)

    def __getitem__(self, key: Interval) -> int:
        return self.daily_demands[key - 1]

    @classmethod
    def empty(cls) -> Demand:
        if cls._empty is None:
            cls._empty = cls(Day(0), 0, [[0, 0], []])
        return cls._empty



# Radiation
_VDI = Union[Vertex, Day, Interval]
class Radiation:
    def __init__(self, c: Circumstances, g: Graph) -> None:
        self.circumstances = c
        self.graph = g
        self._value = [
            [
                [0.] * c.intervals_per_day
                for _ in range(c.number_of_days)
            ] for _ in range(g.v)
        ]


    @overload
    def __getitem__(self, key: Vertex) -> list[list[float]]: ...

    @overload
    def __getitem__(self, key: Day) -> list[list[float]]: ...

    @overload
    def __getitem__(self, key: tuple[Vertex, Day]) -> list[float]: ...

    @overload
    def __getitem__(self, key: tuple[Day, Vertex]) -> list[float]: ...

    @overload
    def __getitem__(self, key: tuple[Vertex, Day, Interval]) -> float: ...

    def __getitem__(self, key: _VDI | tuple[_VDI] | tuple[_VDI, _VDI] | tuple[_VDI, _VDI, _VDI]):
        if   isinstance(key, Vertex):
            return self._value[key.id - 1]
        elif isinstance(key, Day):
            return [l[key - 1] for l in self._value]
        elif isinstance(key, Interval):
            return [[r[key - 1] for r in l] for l in self._value]
        else:
            l = len(key)
            if   l == 1:
                k = key[0]
                if   isinstance(k, Vertex):
                    return self._value[k.id - 1]
                elif isinstance(k, Day):
                    return [l[k - 1] for l in self._value]
                else:
                    return [[r[k - 1] for r in l] for l in self._value]
            elif l == 2:
                k, e = key[0], key[1]
                if   isinstance(k, Vertex):
                    if   isinstance(e, Day):
                        return self._value[k.id - 1][e - 1]
                    elif isinstance(e, Interval):
                        return [r[e - 1] for r in self._value[k.id - 1]]
                elif isinstance(k, Day):
                    if   isinstance(e, Vertex):
                        return self._value[e.id - 1][k - 1]
                    elif isinstance(e, Interval):
                        return [l[k - 1][e - 1] for l in self._value]
                else:
                    if   isinstance(e, Vertex):
                        return [r[k - 1] for r in self._value[e.id - 1]]
                    elif isinstance(e, Interval):
                        return [l[e - 1][k - 1] for l in self._value]
            elif l == 3:
                f: Callable[[Any], int] = lambda x:\
                    1 * isinstance(x, Vertex)\
                  + 2 * isinstance(x, Day)\
                  + 4 * isinstance(x, Interval)
                if f(key) == 7:
                    k, e, y = sorted(key, key=f)
                    if isinstance(k, Vertex) and isinstance(e, Day) and isinstance(y, Interval):
                        return self._value[k.id - 1][e - 1][y - 1]
        raise KeyError


    def __setitem__(self, key: tuple[Day, Vertex], value: list[float]) -> None:
        k, e = key
        self._value[e.id - 1][k - 1] = value
        


# Assets Classes
class AssetCatalog:
    def __init__(self, info: list[list[int]]) -> None:
        (
            (self.N_PV,),
            (self.N_FE,),
            (self.N_RB,),
            (self.N_EVC,),
            (self.N_V,),
        ) = info

        self.PV = PV; self.FE = FE; self.RB = RB; self.EVC = EVC
        self.EV = Vehicle  # kouiuno ammari yokunai to omou yo
        


class AbsAsset:
    variety: ClassVar[list[Any]]
    variety_id: int
    C_init: int

    @property
    def initial_cost(self) -> int:
        return self.C_init

    def __str__(self) -> str:
        return str(self.variety_id)

    @classmethod
    def add_variety(cls, *v: int) -> None:
        cls.variety.append(v)


class PV(AbsAsset):
    variety: ClassVar[list[tuple[int, int]]] = []

    variety_id: int; unit_area: int; C_init: int
    amount: int
    
    def __new__(cls, variety_id: int, amount: int) -> PV:
        s = AbsAsset.__new__(cls)
        s.variety_id = variety_id
        s.unit_area, s.C_init = cls.variety[variety_id - 1]
        s.amount = amount
        return s

    @property
    def area(self) -> int:
        return self.unit_area * self.amount

    @property
    def initial_cost(self) -> int:
        return self.C_init * self.amount


class FE(AbsAsset):
    variety: ClassVar[list[tuple[int, int]]] = []
    
    variety_id: int; P_max: int; C_init: int

    def __new__(cls, variety_id: int) -> FE:
        s = AbsAsset.__new__(cls)
        s.variety_id = variety_id
        s.P_max, s.C_init = cls.variety[variety_id - 1]
        return s


class RB(AbsAsset):
    variety: ClassVar[list[tuple[int, int]]] = []
    
    variety_id: int; unit_Cap: int; C_init: int
    amount: int

    def __new__(cls, variety_id: int, amount: int) -> RB:
        s = AbsAsset.__new__(cls)
        s.variety_id = variety_id
        *_, s.unit_Cap, s.C_init = cls.variety[variety_id - 1]
        s.amount = amount
        return s

    @property
    def total_Cap(self) -> int:
        return self.unit_Cap * self.amount

    @property
    def initial_cost(self) -> int:
        return self.C_init * self.amount


class EVC(AbsAsset):
    variety: ClassVar[list[tuple[int, int, int]]] = []
    
    variety_id: int; P_in: int;P_out: int; C_init: int

    def __new__(cls, variety_id: int) -> EVC:
        s = AbsAsset.__new__(cls)
        s.variety_id = variety_id
        s.P_in, s.P_out, s.C_init = cls.variety[variety_id - 1]
        return s


class Vehicle(AbsAsset):
    variety: ClassVar[list[tuple[int, int, int, int, int, int]]] = []
    graph: ClassVar[Graph]

    variety_id: int
    Cap_charge: int; Cap_order: int
    P_charge: int; P_discharge: int; C_init: int; Delta_move: int

    graph: Graph
    vertex: Vertex; Chg_init: int
    Chg_current: int; pos: tuple[Vertex, Vertex, int]

    def __new__(cls, variety_id: int, v: Vertex, chg: int) -> Vehicle:
        s = AbsAsset.__new__(cls)
        s.variety_id = variety_id
        (
            s.Cap_charge, s.Cap_order,
            s.P_charge, s.P_discharge, s.C_init, s.Delta_move
        ) = cls.variety[variety_id]
        
        s.vertex, s.Chg_init = v, chg
        return s

    def __str__(self) -> str:
        return f"{self.vertex} {self.Chg_init} {self.variety_id}"

    def is_valid(self) -> bool:
        return self.Chg_init <= self.Cap_charge

    def return_distance(self) -> int:
        u, v, d = self.pos; c = self.graph.get_edge_dist(u, v) - d
        value = min(
            d + self.graph.distance(u, self.vertex),
            c + self.graph.distance(v, self.vertex)
        )
        return value



# Transportation Request
class OrderFrequency:
    def __init__(self, c: Circumstances) -> None:
        self.circumstance = c
        self._frequency = [
            [0] * c.intervals_per_day for _ in range(c.number_of_days)
        ]

    @overload
    def __getitem__(self, key: Day) -> list[int]: ...

    @overload
    def __getitem__(self, key: tuple[Day, Interval]) -> int: ...

    def __getitem__(self, key: Day | Interval | tuple[Day | Interval] | tuple[Day, Interval]):
        if isinstance(key, Day):
            return self._frequency[key - 1]
        elif isinstance(key, Interval):
            return [l[key - 1] for l in self._frequency]
        else:
            if (l := len(key)) == 1:
                k = key[0]
                if isinstance(k, Day):
                    return self._frequency[k - 1]
                else:
                    return [l[k - 1] for l in self._frequency]
            elif l == 2:
                day, interval = key[0], key[1]
                if isinstance(day, Day) and isinstance(interval, Interval):
                    return self._frequency[day - 1][interval - 1]
        raise KeyError


    def __setitem__(self, key: Day, value: list[int]):
        self._frequency[key - 1] = value



class Order(NamedTuple):
    id: int
    ordered: Step; arrived: Step
    v_pickup: Vertex
    v_destination: Vertex
    state: int  # 0, 1, 2 respectively indicate UNSENT, IN_TRANSIT, DELIVERED

    @property
    def time_spent(self) -> int:
        return self.arrived - self.ordered



# Shelter
class Shelters:
    def __init__(self, info: tuple[int, list[list[int]], list[int]], c: Circumstances, g: Graph) -> None:
        self.circumstances = c
        self.graph = g
        self.N, xp, self.D = info
        self._value = [Shelter(g.getvertex(x), p) for x, p in xp]
        self.vertex: defaultdict[Vertex, list[Shelter]] = defaultdict(list)
        for v in self._value:
            self.vertex[v.vertex].append(v)

    def predicted_demand(self, v: Vertex, i: Interval) -> int:
        l = [s.pop * self.D[i - 1] // 100 for s in self.vertex[v]]
        return sum(l)


class Shelter(NamedTuple):
    vertex: Vertex
    pop: int



# Answers and Results
class NanoGrid(NamedTuple):
    vertex: Vertex; Chg_init: int
    pv: PV; fe: FE; rb: RB; evc: EVC

    Chg_current: int = 0


    def __str__(self) -> str:
        s = f"{self.vertex} {self.Chg_init}\n"\
            f"{self.pv} {self.pv.amount}\n"\
            f"{self.fe}\n"\
            f"{self.rb} {self.rb.amount}\n"\
            f"{self.evc}"
        return s


    @property
    def initial_cost(self) -> int:
        value = self.pv.initial_cost\
              + self.fe.initial_cost\
              + self.rb.initial_cost\
              + self.evc.initial_cost
        return value


    def is_valid(self) -> bool:
        value = self.pv.area <= self.vertex.area\
            and self.Chg_init <= self.rb.total_Cap
        return value



class Answer:
    def __init__(
        self,
        grids: list[NanoGrid] | None = None,
        evs: list[Vehicle] | None = None,
    ) -> None:
        self.grids = grids or []
        self.evs = evs or []


    def __str__(self) -> str:
        f: Callable[[Any], str] = lambda i: "\n".join(map(str, i))
        lg, le = len(self.grids), len(self.evs)
        if lg > 0 and le > 0:
            return f((lg, f(self.grids), le, f(self.evs)))
        elif lg > 0:
            return f((lg, f(self.grids), le))
        elif le > 0:
            return f((lg, le, f(self.evs)))
        else:
            return f((lg, le))
        

    @property
    def initial_cost(self) -> int:
        value = sum(g.initial_cost for g in self.grids)\
              + sum(ev.initial_cost for ev in self.evs)
        return value


    def add_grid(self, grid: NanoGrid) -> None:
        self.grids.append(grid)

    def add_EV(self, ev: Vehicle) -> None:
        self.evs.append(ev)

    def is_valid(self) -> bool:
        return all(g.is_valid() for g in self.grids)\
           and all(ev.is_valid() for ev in self.evs)



class Result:  # almost a container of unknown values of the scoring function
    def orders_Tmax(self, day: Day) -> list[Order]: ...

    def grids_Tmax(self, day: Day) -> list[NanoGrid]: ...
    
    def evs_Tmax(self, day: Day) -> list[Vehicle]: ...

    def total_FE_power_generation(self, day: Day) -> int: ...

    def total_power_purchase(self, day: Day) -> int: ...

    def uninterruptible_days(self) -> int: ...




## Functions
_ScoreInfo = Tuple[List[int], List[int], List[float], List[float], List[float], List[float], List[int]]

@overload
def io_1(query: Literal["budget"]) -> int: ...

@overload
def io_1(query: Literal["temporal"]) -> list[int]: ...

@overload
def io_1(query: Literal["score"]) -> _ScoreInfo: ...

@overload
def io_1(query: Literal["graph"]) -> tuple[list[int], list[list[int]], list[list[int]]]: ...

@overload
def io_1(query: Literal["demand"]) -> int: ...

@overload
def io_1(q: Literal["demand"], u: int, ery: int) -> list[list[int]]: ...

@overload
def io_1(q: Literal["radiation"], u: int, ery: int) -> list[float]: ...

@overload
def io_1(query: Literal["asset"]) -> list[list[int]]: ...

@overload
def io_1(q: Literal["asset"], u: Literal["PV", "FE", "RB", "EVC"], ery: int) -> list[int]: ...

@overload
def io_1(q: Literal["asset"], u: Literal["vehicle"], ery: int) -> list[list[int]]: ...

@overload
def io_1(q: Literal["order"], uery: int) -> list[int]: ...

@overload
def io_1(query: Literal["shelter"]) -> tuple[int, list[list[int]], list[int]]: ...

@overload
def io_1(query: Literal["end"]) -> int: ...

def io_1(*query: Any):
    IO.print(" ".join(map(str, query)))

    q, *uery = query
    if   q == "budget":
        value = IO.int0d()
    elif q == "temporal":
        value = IO.int1d()
    elif q == "score":
        value = (
            IO.int1d(),
            IO.int1d(),
            IO.float1d(),
            IO.float1d(),
            IO.float1d(),
            IO.float1d(),
            IO.int1d(),
        )
    elif q == "graph":
        V, E = IO.int1d()
        vertices = IO.int2d(V)
        edges = IO.int2d(E)
        value = ([V, E], vertices, edges)
    elif q == "demand":
        l = len(uery)
        if l == 0:
            value = IO.int0d()
        elif l == 2:
            value = IO.int2d(2)
        else:
            raise ValueError
    elif q == "radiation":
        value = IO.float1d()
    elif q == "asset":
        l = len(uery)
        if   l == 0:
            value = IO.int2d(5)
        elif l == 2:
            u, _ = uery
            if   u == "PV":
                value = IO.int1d()
            elif u == "FE":
                value = IO.int1d()
            elif u == "RB":
                value = IO.int1d()
            elif u == "EVC":
                value = IO.int1d()
            elif u == "vehicle":
                value = IO.int2d(2)
            else:
                raise ValueError
        else:
            raise ValueError
    elif q == "order":
        value = IO.int1d()
    elif q == "shelter":
        N = IO.int0d()
        xp = IO.int2d(N)
        d = IO.int1d()
        value = (N, xp, d)
    elif q == "end":
        value = 0
    else:
        raise ValueError

    return value



def io_2(
    answer: Answer, *, command: str, day: Day=Day(0), opt: int
) -> tuple[float, float, float] | None:
    IO.print(answer)
    
    IO.print(command, day, opt)
    if   command == "test":
        c0, c1, c2 = IO.float1d()
        return (c0, c1, c2)
    elif command == "submit":
        return None
    else:
        raise ValueError('argument "command" must be "test" or "submit"')




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
    ### write logic here ########
    # grids = [NanoGrid(graph.getvertex(1), 0, PV(1, 1), FE(1), RB(1, 1), EVC(1))]
    # evs = [Vehicle(1, graph.getvertex(1), 0)]
    # ans = Answer(grids, evs)
    # io_2(ans, command="submit", opt=0)
    #############################
    pass




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
    
    order_freq = OrderFrequency(circumstances)
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








