{.pragma: onceCheck.}
{.
  passC:
    "-std=gnu++17 -x c++ -I/home/casper/micromamba/envs/boiler/lib/python3.12/site-packages/include/ -I/usr/include"
.}
{.
  passL:
    "-lstdc++ -L/home/casper/micromamba/envs/boiler/lib/python3.12/site-packages -l:libnetworkit.so -Wl,-rpath,/home/casper/micromamba/envs/boiler/lib/python3.12/site-packages"
.}
import strformat, math, tables, sequtils, sets
from boiler_room.agent import State, Agent, toAdj

type
  Graph* {.importcpp: "NetworKit::Graph", header: "<networkit/graph/Graph.hpp>".} = object

  Betweenness {.
    importcpp: "NetworKit::Betweenness",
    header: "<networkit/centrality/Betweenness.hpp>"
  .} = object

  Closeness {.
    importcpp: "NetworKit::Closeness", header: "<networkit/centrality/Closeness.hpp>"
  .} = object

  DegreeCentrality {.
    importcpp: "NetworKit::DegreeCentrality",
    header: "<networkit/centrality/DegreeCentrality.hpp>"
  .} = object
  Vector {.importcpp: "std::vector<double>", header: "<vector>".} = object

  ClosenessVariant {.
    importcpp: "NetworKit::ClosenessVariant",
    header: "<networkit/centrality/Closeness.hpp>"
  .} = enum
    Standard
    Generalized

  NodeMap {.importcpp: "std::map<NetworKit::node, double>", header: "<map>".} = object

  Edge {.importcpp: "NetworKit::Edge", header: "<networkit/graph/Graph.hpp>".} = object

  EdgeIterator {.
    importcpp: "NetworKit::Graph::EdgeIterator", header: "<networkit/graph/Graph.hpp>"
  .} = object

  EdgeRange {.
    importcpp: "NetworKit::Graph::EdgeRange", header: "<networkit/graph/Graph.hpp>"
  .} = object

  NodeIterator {.
    importc: "NetworKit::Graph::NodeIterator", header: "<networkit/graph/Graph.hpp"
  .} = object

  NodeSet {.
    importcpp: "std::unordered_set<NetworKit::node>", header: "<unordered_set>"
  .} = object

  NeighborRange {.
    importcpp: "NetworKit::Graph::NeighborRange", header: "networkit/graph/Graph.hpp"
  .} = object

  NeighborIterator {.
    importcpp: "NetworKit::Graph::NeighborIterator",
    header: "<networkit/graph/Graph.hpp>"
  .} = object

  NodeVector {.importcpp: "std::vector<NetworKit::node>", header: "<vector>".} = object

proc newNodeVector(): NodeVector {.
  importcpp: "std::vector<NetworKit::node>()", constructor
.}

proc add(v: var NodeVector, node: int) {.importcpp: "#.push_back(@)".}
proc len(v: NodeVector): int {.importcpp: "#.size()".}
proc `[]`(v: NodeVector, i: int): int {.importcpp: "#[#]".}

proc newGraph*(nodes: int): Graph {.importcpp: "NetworKit::Graph(@)", constructor.}
proc addEdge*(g: var Graph, u, v: int) {.importcpp: "#.addEdge(@, @)".}
proc numberOfNodes*(g: Graph): int {.importcpp: "#.numberOfNodes()".}

proc numberOfEdges*(g: Graph): int {.importcpp: "#.numberOfEdges()".}

proc u*(e: Edge): int {.importcpp: "#.u".}
proc v*(e: Edge): int {.importcpp: "#.v".}

proc newBetweenness(
  g: Graph
): Betweenness {.importcpp: "NetworKit::Betweenness(@)", constructor.}

proc newCloseness(
  g: Graph, normalized: bool, variant: ClosenessVariant
): Closeness {.importcpp: "NetworKit::Closeness(@)", constructor.}

proc newDegreeCentrality(
  g: Graph
): DegreeCentrality {.importcpp: "NetworKit::DegreeCentrality(@)", constructor.}

proc run(b: var Betweenness) {.importcpp: "#.run()".}
proc run(c: var Closeness) {.importcpp: "#.run()".}
proc run(d: var DegreeCentrality) {.importcpp: "#.run()".}

proc scores(b: Betweenness): Vector {.importcpp: "#.scores()".}
proc scores(c: Closeness): Vector {.importcpp: "#.scores()".}
proc scores(d: DegreeCentrality): Vector {.importcpp: "#.scores()".}

proc newNodeSet(): NodeSet {.
  importcpp: "std::unordered_set<NetworKit::node>()", constructor
.}

proc insert*(s: var NodeSet, node: int) {.importcpp: "#.insert(@)".}

proc edgeRange*(g: Graph): EdgeRange {.importcpp: "#.edgeRange()".}

#proc begin(it: EdgeRange): EdgeIterator {.importcpp: "#.begin()".#}
#proc `end`(it: EdgeRange): EdgeIterator {.importcpp: "#.end()".}
#

proc begin(it: EdgeIterator): EdgeIterator {.importcpp: "#.begin()".}

proc `end`(it: EdgeIterator): EdgeIterator {.importcpp: "#.end()".}
proc `!=`*(a, b: EdgeIterator): bool {.importcpp: "(# != #)".}
proc `*`*(it: EdgeIterator): Edge {.importcpp: "*#".}
proc inc*(it: var EdgeIterator) {.importcpp: "++#".}
iterator items(n: EdgeIterator): int =
  var it = n.begin()
  let endIt = n.`end`()
  while it != endIt:
    yield (`*`(it)).v
    it.inc

proc begin*(range: EdgeRange): EdgeIterator {.importcpp: "#.begin()".}
proc `end`*(range: EdgeRange): EdgeIterator {.importcpp: "#.end()".}
iterator edges*(g: Graph): Edge =
  let edges = g.edgeRange()
  var it = edges.begin()
  let endIt = edges.end()
  var idx = 0
  while it != endIt:
    let edge = `*` it
    yield edge
    it.inc

proc neighborRange*(
  g: Graph, node: int
): NeighborRange {.importcpp: "#.neighborRange(@)".}

proc begin*(range: NeighborRange): NeighborIterator {.importcpp: "#.begin()".}
proc `end`*(range: NeighborRange): NeighborIterator {.importcpp: "#.end()".}

proc `!=`*(a, b: NeighborIterator): bool {.importcpp: "(# != #)".}
proc `*`*(it: NeighborIterator): int {.importcpp: "*#".}
proc inc*(it: var NeighborIterator) {.importcpp: "++#".}

iterator neighbors*(g: Graph, node: int): int =
  let neighborRange = g.neighborRange(node)
  var it = neighborRange.begin()
  let endIt = neighborRange.`end`()
  while it != endIt:
    yield `*` it
    inc(it)

proc removeEdge*(g: var Graph, i, j: int) {.importcpp: "#.removeEdge(#, #)".}

proc addNode*(g: var Graph, node: int) {.importcpp: "#.addNode(@)".}
proc removeNode*(g: var Graph, node: int) {.importcpp: "#.removeNode(@)".}
proc hasNode*(g: Graph, node: int): bool {.importcpp: "#.hasNode(@)".}
proc upperNodeIdBound*(g: Graph): int {.importcpp: "#.upperNodeIdBound()".}

proc `[]`(v: Vector, i: int): float {.importcpp: "#[#]".}

proc `[]`(m: NodeMap, key: int): float {.importcpp: "#[#]".}
proc `[]=`(m: var NodeMap, key: int, val: float) {.importcpp: "#[#] = #".}

proc subgraphFromNodes*(
  G: Graph, nodes: openArray[int], compact: bool = false
): Graph {.
  importcpp:
    """
  [&](const auto& G, auto nodes, bool compact) {
    return NetworKit::GraphTools::subgraphFromNodes(G, nodes.begin(), nodes.end(), compact);
  }(#, #, #)
"""
.}

# Alternative version using HashSet
proc subgraphFromNodesSet*(
  G: Graph, nodes: NodeSet, compact: bool = false
): Graph {.
  importcpp:
    """
  [&](const auto& G, const auto& nodes, bool compact) {
    return NetworKit::GraphTools::subgraphFromNodes(G, nodes.begin(), nodes.end(), compact);
  }(#, #, #)
""",
  header: "<networkit/graph/GraphTools.hpp>"
.}

proc getEgoNetwork*(g: Graph, ego: int): Graph =
  var egoNetworkNodes = newNodeSet()
  egoNetworkNodes.insert(ego)

  for neighbor in g.neighbors(ego):
    egoNetworkNodes.insert(neighbor)

  result = subgraphFromNodesSet(g, egoNetworkNodes)

proc toGraph*(s: State): Graph =
  result = newGraph(s.agents.len)
  for (i, j) in s.toAdj():
    result.addEdge(i, j)

proc getBetweenness*(s: State): Table[int, float] =
  var g = s.toGraph
  var betweenness = newBetweenness(g)
  betweenness.run()
  let betweennessScores = betweenness.scores()
  result = initTable[int, float]()
  for node in (0 ..< g.numberOfNodes()):
    result[node] = betweennessScores[node].float

proc getCloseness*(s: State): Table[int, float] =
  var g = s.toGraph
  var closeness = newCloseness(g, true, ClosenessVariant.Standard)
  closeness.run()
  let closenessScores = closeness.scores()
  result = initTable[int, float]()
  for node in (0 ..< g.numberOfNodes()):
    result[node] = closenessScores[node].float

proc getDegreeCentrality*(s: State): Table[int, float] =
  var g = s.toGraph
  var degreeCentrality = newDegreeCentrality(g)
  degreeCentrality.run()
  let degreeCentralityScores = degreeCentrality.scores()
  result = initTable[int, float]()
  for node in (0 ..< g.numberOfNodes()):
    result[node] = degreeCentralityScores[node].float

proc getAttributeAssorativity*(s: State): Table[int, float] =
  var g = s.toGraph
  var attributes: NodeMap
  var roleMap = initTable[string, float]()
  for idx, role in s.config.p_roles.keys.toSeq():
    roleMap[role] = idx.float

  var states = newSeq[float](s.agents.len)
  for agent in s.agents:
    let role = roleMap[agent.role]
    attributes[agent.id] = role
    states[agent.id] = agent.state

  proc catAssortativity(g: Graph, center: int, states: seq[float]): float =
    let ego = g.getEgoNetwork(center)
    let m = g.numberOfEdges().float
    var ai, bi: Table[int, float64]
    var eii = 0.0
    for edge in ego.edges():
      if states[edge.u.int] == 0.0 or states[edge.v().int] == 0.0:
        continue
      let x = attributes[edge.u()].int
      let y = attributes[edge.v()].int
      if x == y:
        eii += 1.0
      ai.mgetOrPut(x, 0.0) += 0.5
      ai.mgetOrPut(y, 0.0) += 0.5
      bi.mgetOrPut(x, 0.0) += 0.5
      bi.mgetOrPut(y, 0.0) += 0.5

    eii /= m
    var a = 0.0
    for k, v in ai:
      a += v * bi[k]
    a /= m * m
    result = (eii - a) / (1.0 - a)

  result = initTable[int, float]()
  for node in (0 ..< g.numberOfNodes()):
    result[node] = catAssortativity(g, node, states)
