from __future__ import annotations

from collections import defaultdict
from itertools import product
from statistics import mean
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
    def errprint(*i: Any, **kwargs: Any) -> None:
        from sys import stderr
        print(*i, **kwargs, file=stderr)



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

    def __repr__(self) -> str:
        return f"Position{tuple(self._value)}"

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
    def __getitem__(self, key: Day | tuple[Day]) -> dict[int, Demand]: ...

    @overload
    def __getitem__(self, key: int | tuple[int]) -> dict[int, Demand]: ...

    @overload
    def __getitem__(self, key: tuple[Day, int]) -> Demand: ...

    @overload
    def __getitem__(self, key: tuple[Day, int, Interval]) -> int: ...

    def __getitem__(self, key: Day | int | tuple[Day | int] | tuple[Day, int] | tuple[Day, int, Interval]):
        if isinstance(key, Day):
            return {i: v for i, v in enumerate(self._value[key - 1], 1)}
        elif isinstance(key, int):
            return {day: self._value[day - 1][key - 1] for day in self.circumstances.days()}
        else:
            l = len(key)
            if   l == 1:
                k = key[0]
                if isinstance(k, Day):
                    return {i: v for i, v in enumerate(self._value[k - 1], 1)}
                else:
                    return {day: self._value[day - 1][k - 1] for day in self.circumstances.days()}
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


    @property
    def vertices(self) -> list[Vertex]:
        return [d.vertex for d in self._value[Day(0)]]


class Demand:
    circumstances: ClassVar[Circumstances]
    graph: ClassVar[Graph]
    _revision: ClassVar[list[list[float]]] = [
        [1.8241241314072116, 1.882833150107928,  1.9391623863209513, 1.99337826079715,   2.0457011604472948, 2.096315802221852,  2.145378792894321,  2.193024254736208,  2.239368086819472,  2.2845112442856284, 2.328542297893525,  2.371539457440204,  2.41357218989912,   2.454702527070049,  2.4949861324527074, 2.5344731793163824, 2.573209079201291,  2.611235090815617,  2.648588832456274,  2.6853047159815526, 2.7214143165174844],
        [2.5057938380690183, 2.5566797889782125, 2.606115072685674,  2.6541959268152517, 2.7010131637113517, 2.746650885445857,  2.7911862296052243, 2.8346896096897707, 2.877225176253037,  2.9188513581696216, 2.9596214126967855, 2.9995839495513716, 3.0387834136271854, 3.0772605212178963, 3.1150526498909468, 3.152194184737265,  3.1887168248327997, 3.2246498540721595, 3.2600203804503805, 3.2948535475867127, 3.329172721920354 ],
        [3.290237596434382,  3.330333618811209,  3.3701033531097693, 3.4094634159938355, 3.448362066057797,  3.4867688290307797, 3.524667671488613,  3.5620524272597276, 3.598923685122738,  3.6352866428704527, 3.6711496114869737, 3.706522963375262,  3.741418387982561,  3.7758483627467094, 3.8098257764232955, 3.8433636612087776, 3.8764750031227755, 3.9091726090285106, 3.9414690148379785, 3.9733764237634377, 4.004906666527014 ],
        [4.155051839760759,  4.183775343346494,  4.213104285476907,  4.2428475366890535, 4.272858936066906,  4.303026036287938,  4.333261882350746,  4.363498962191104,  4.393684716679534,  4.423778176568186,  4.453747420627425,  4.483567637565257,  4.51321963595863,   4.542688689622601,  4.571963636328729,  4.6010361694738275, 4.629900277866243,  4.658551800066672,  4.686988067951546,  4.715207620229995,  4.743209971148309 ],
        [5.076829040319717,  5.095537480806867,  5.115365983816718,  5.136114437380213,  5.157616117009374,  5.179732298619751,  5.202347473221811,  5.22536528482832,   5.2487051596698695, 5.272299540637808,  5.296091629859681,  5.320033548780216,  5.344084837484699,  5.368211228227246,  5.392383640207285,  5.416577352938432,  5.440771324033441,  5.464947624076518,  5.489090966726744,  5.513188316546607,  5.53722856050061  ],
        [6.035196786992038,  6.0462768449982365, 6.058567109616888,  6.071928993578439,  6.086234646176703,  6.101368464366112,  6.117227175050419,  6.133719189196401,  6.150763624980037,  6.168289219998606,  6.18623325020223,   6.204540514617156,  6.223162411861375,  6.242056116187069,  6.2611838510731905, 6.280512253585816,  6.300011820692365,  6.319656428215362,  6.339422913419884,  6.359290712930605,  6.379241548533505 ],
        [7.014866642992825,  7.02083424423151,   7.027818890696997,  7.035763248167214,  7.04460238967833,   7.054268731998611,  7.064695177259327,  7.0758170240591065, 7.087573055122022,  7.0999060847584525, 7.112763159114556,  7.126095538902777,  7.139858550852413,  7.154011364649011,  7.168516732294865,  7.183340713542327,  7.198452402202289,  7.213823662264829,  7.229428878908829,  7.245244726962355,  7.261249957761457 ],
        [8.005776075719512,  8.008699343691971,  8.012339162179044,  8.016700179597672,  8.021772431814721,  8.027535676704227,  8.033962810467319,  8.041022460607802,  8.048680900931243,  8.056903433873956,  8.065655366240327,  8.074902680549725,  8.084612481745575,  8.094753280075354,  8.105295155816062,  8.116209839800813,  8.127470734799283,  8.139052896122147,  8.150932984843047,  8.163089203345294,  8.175501220180193 ],
        [9.002060113671224,  9.003362724996293,  9.005102180662938,  9.007312667944198,  9.010016030962896,  9.013223641243867,  9.016938353447268,  9.021156323405839,  9.025868592974343,  9.031062417346009,  9.036722346727103,  9.042831090248246,  9.049370194918936,  9.056320571739452,  9.06366289785024,   9.071377919488139,  9.079446676379549,  9.08785066442533,   9.096571950266666,  9.105593248585656,  9.114897970749592 ],
        [10.000673355312507, 10.00120145159635,  10.00196387723698,  10.002998535336973, 10.004336819341603, 10.006003400715166, 10.008016548716514, 10.010388742905363, 10.01312741431296,  10.016235711643573, 10.019713232231924, 10.023556686464303, 10.027760482693656, 10.032317230639224, 10.037218167377635, 10.042453513084148, 10.04801276487054,  10.053884937170656, 10.060058756641762, 10.066522818783195, 10.073265712605478],
        [11.000201398893536, 11.00039622452629,  11.000702766553971, 11.001150022421752, 11.001765430728986, 11.002573828338663, 11.003596854420763, 11.004852703069151, 11.006356124363034, 11.008118588511138, 11.010148547031257, 11.012451743134815, 11.01503153851246,  11.017889235209742, 11.021024379635662, 11.024435041582672, 11.028118065067995, 11.032069290355123, 11.036283748089254, 11.040755827387114, 11.045479420181668],
        [12.00005505360479,  12.00012047263581,  12.000233544469275, 12.000412115655644, 12.000675011311326, 12.001041113734416, 12.00152861726819,  12.00215448125128,  12.002934068144857, 12.003880937718423, 12.005006763612066, 12.00632134033903,  12.007832653362714, 12.00954699025238,  12.011469076046705, 12.013602220379061, 12.015948467516198, 12.018508743268566, 12.021282994852173, 12.024270321343298, 12.027469093487882],
        [13.000013739259531, 13.000033737166993, 13.000072007257673, 13.00013786588165,  13.000242204552244, 13.000397010303985, 13.000614866678118, 13.000908492943953, 13.001290353359426, 13.001772348614729, 13.002365588733321, 13.003080239432908, 13.003925430649508, 13.004909215155546, 13.006038565872968, 13.007319401860084, 13.008756634578141, 13.010354227656777, 13.012115264841993, 13.014042022069226, 13.016136040647163],
        [14.000003127428045, 14.000008694044464, 14.000020581227261, 14.000043020740405, 14.00008149663977,  14.000142620188488, 14.000233923673685, 14.000363611798875, 14.000540303177413, 14.000772785315412, 14.001069797543067, 14.001439849080048, 14.001891074164176, 14.002431122749252, 14.003067083301536, 14.003805433288909, 14.00465201271389,  14.005612016223877, 14.006689999748444, 14.007889898135671, 14.009215050806059],
        [15.0000006487982,   15.000002060145341, 15.00000544929774,  15.000012513626103, 15.000025697905643, 15.00004823506945,  15.000084123451456, 15.000138051578423, 15.000215286214692, 15.00032153966083,  15.000462830085462, 15.000645345414965, 15.000875317976295, 15.001158914186986, 15.001502141303751, 15.001910771585239, 15.002390283114922, 15.002945815855467, 15.003582141154837, 15.004303642800881, 15.005114307748011],
        [16.00000012258235,  16.00000044858899,  16.0000013357008,   16.00000339086005,  16.000007589352375, 16.000015349921068, 16.000028581033728, 16.00004969457379,  16.000081588749055, 16.000127605538697, 16.000191469592362, 16.000277215629392, 16.00038911062111,  16.000531575841546, 16.00070911256831,  16.00092623400067,  16.00118740493216,  16.00149698990144,  16.001859209939994, 16.002278107611883, 16.002757519766178],
        [17.000000021080393, 17.00000008970612,  17.000000302925482, 17.000000855512575, 17.000002098158312, 17.000004594061693, 17.00000916960799,  17.000016952830553, 17.000029395991504, 17.000048281278055, 17.000075710725,    17.000114082841804, 17.000166059087967, 17.000234523461103, 17.000322538220505, 17.00043329832586,  17.00057008664742,  17.000736231481078, 17.000935067427363, 17.00116990029054,  17.001443976329153],
        [18.000000003297853, 18.000000016466334, 18.000000063534223, 18.000000200876045, 18.00000054275176,  18.000001292542645, 18.00000277684818,  18.000005478527786, 18.00001006510493,  18.00001740990132,  18.000028604450343, 18.000044961856542, 18.00006801165258,  18.000099487300936, 18.00014130780916,  18.000195555027734, 18.000264448141063, 18.000350316703912, 18.000455573365574, 18.000582687197138, 18.00073415831697 ],
        [19.00000000046912,  19.000000002773177, 19.000000012317905, 19.000000043876675, 19.000000131316263, 19.000000341728345, 19.000000793447548, 19.00000167654528,  19.00000327391294,  19.00000598099008,  19.000010322471795, 19.00001696480496,  19.000026723818706, 19.00004056733182,  19.000059612980525, 19.000085121794378, 19.000118488218593, 19.00016122735594,  19.000214960202488, 19.00028139760138,  19.000362323557734],
        [20.000000000060652, 20.000000000428344, 20.000000002206768, 20.00000000891214,  20.0000000297053,   20.000000084869953, 20.00000021384662,  20.000000485684453, 20.000001011338224, 20.000001956927964, 20.00000355694525,  20.000006126426882, 20.00001007127838,  20.000015896161845, 20.000024209613812, 20.00003572629216,  20.00005126644518,  20.00007175284066,  20.00009820548885,  20.000131734545267, 20.00017353179619 ],
        [21.000000000007123, 21.00000000006066,  21.00000000036519,  21.000000001682785, 21.0000000062807,   21.00000001979376,  21.000000054346685, 21.000000133153538, 21.000000296607258, 21.000000609649856, 21.000001170027556, 21.000002116920054, 21.00000363939779,  21.000005984196452, 21.000009462384035, 21.00001445460945,  21.0000214147439,   21.00003087184089,  21.000043430439593, 21.00005976931365,  21.000080638823366],
        [22.00000000000076,  22.000000000007876, 22.000000000055806, 22.00000000029529,  22.00000000124084,  22.000000004333934, 22.000000013020028, 22.00000003453787,  22.00000008256783,  22.000000180792536, 22.000000367310758, 22.000000699739303, 22.00000126075747,  22.000002163803202, 22.000003558621852, 22.00000563639048,  22.00000863418505,  22.000012838614612, 22.000018588507825, 22.000026276595783, 22.00003635018735 ],
        [23.000000000000075, 23.000000000000934, 23.000000000007876, 23.000000000048143, 23.000000000229004, 23.00000000089065,  23.000000002939764, 23.000000008473805, 23.000000021811328, 23.000000051024145, 23.0000001100254,   23.000000221211653, 23.00000041859893,  23.00000075134172,  23.000001287487752, 23.000002117802172, 23.000003359492883, 23.00000515968116,  23.000007698483973, 23.0000111916038,   23.00001589235263 ],
        [24.000000000000004, 24.0000000000001,   24.000000000001027, 24.00000000000729,  24.00000000003947,  24.000000000171752, 24.000000000625427, 24.000000001966107, 24.000000005466426, 24.00000001370175,  24.000000031440248, 24.000000066870335, 24.000000133181793, 24.000000250485435, 24.000000448026253, 24.00000076661981,  24.000001261224217, 24.000002003551202, 24.000003084619088, 24.000004617156904, 24.000006737780346],
        [25.000000000000004, 25.00000000000001,  25.00000000000012,  25.000000000001023, 25.000000000006352, 25.000000000031072, 25.00000000012535,  25.000000000431317, 25.000000001299544, 25.000000003500254, 25.000000008569046, 25.00000001932565,  25.00000004059708,  25.000000080163602, 25.00000014993012,  25.000000267308277, 25.000000456778693, 25.00000075158998,  25.000001195542108, 25.0000018447971,   25.000002769659652],
    ]
    _empty: ClassVar[Demand | None] = None

    def __init__(self, day: Day, id: int, info: list[list[int]]) -> None:
        self.day = day
        self.id  = id
        (
            (x, self.sigma2),
            self.daily_demands
        ) = info

        self.vertex = self.graph.getvertex(x)

    def __getitem__(self, key: Interval) -> float:
        d = self.daily_demands[key - 1]
        if d > 25:
            return d
        else:
            return self._revision[d - 1][self.sigma2 - 10]

    def __repr__(self) -> str:
        return f"Demand(Day: {self.day}, "\
               f"Vertex: {self.vertex.id}, "\
               f"sigma2: {self.sigma2}, "\
               f"{self.daily_demands})"

    @property
    def average(self) -> float:
        return mean(self[i] for i in self.circumstances.intervals())

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
        self.vertices = [g.getvertex(x) for x, _ in xp]
        self._vtoid = {v:i for v, i in zip(self.vertices, range_1idx(self.N))}

    def predicted_demand(self, v: Vertex, i: Interval) -> int:
        return self._value[self._vtoid[v]].pop * self.D[i - 1] // 100


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
    Demand.circumstances = circumstances
    
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








