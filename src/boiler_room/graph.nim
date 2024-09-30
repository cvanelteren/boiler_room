#{.pragma: onceCheck.}
#{.
#  passC:
#    "-std=gnu++17 -x c++ #-I/home/casper/micromamba/envs/boiler/lib/python3.12/site-pac#kages/include/ -I/usr/include"
#.}
#{.
#  passL:
#    "-lstdc++ #-L/home/casper/micromamba/envs/boiler/lib/python3.12/site-pac#kages -l:libnetworkit.so #-Wl,-rpath,/home/casper/micromamba/envs/boiler/lib/python3.12#/site-packages"
#.}
#
#import strformat, math, tables, sequtils, sets
#from boiler_room.agent import State, Agent, toAdj
#
#type
#  Graph* {.importcpp: "NetworKit::Graph", header: "#<networkit/graph/Graph.hpp>".} = object
#
#  NodeAttribute[T] {.
#    importcpp: "Networkit::NodeAttribute<'0>", header: #"networkit/graph/Graph.hpp"
#  .} = object
#
#  Betweenness {.
#    importcpp: "NetworKit::Betweenness",
#    header: "<networkit/centrality/Betweenness.hpp>"
#  .} = object
#
#  Closeness {.
#    importcpp: "NetworKit::Closeness", header: "#<networkit/centrality/Closeness.hpp>"
#  .} = object
#
#  DegreeCentrality {.
#    importcpp: "NetworKit::DegreeCentrality",
#    header: "<networkit/centrality/DegreeCentrality.hpp>"
#  .} = object
#  Vector {.importcpp: "std::vector<double>", header: "<vector>".#} = object
#
#  ClosenessVariant {.
#    importcpp: "NetworKit::ClosenessVariant",
#    header: "<networkit/centrality/Closeness.hpp>"
#  .} = enum
#    Standard
#    Generalized
#
#  NodeMap {.importcpp: "std::map<NetworKit::node, double>", #header: "<map>".} = object
#
#  NodeRange {.
#    importcpp: "NetworKit::Graph::NodeRange", header: "#<networkit/graph/Graph.hpp>"
#  .} = object
#
#  Edge {.importcpp: "NetworKit::Edge", header: "#<networkit/graph/Graph.hpp>".} = object
#
#  EdgeIterator {.
#    importcpp: "NetworKit::Graph::EdgeIterator", header: "#<networkit/graph/Graph.hpp>"
#  .} = object
#
#  EdgeRange {.
#    importcpp: "NetworKit::Graph::EdgeRange", header: "#<networkit/graph/Graph.hpp>"
#  .} = object
#
#  NodeIterator {.
#    importc: "NetworKit::Graph::NodeIterator", header: "#<networkit/graph/Graph.hpp"
#  .} = object
#
#  NodeSet {.
#    importcpp: "std::unordered_set<NetworKit::node>", header: "#<unordered_set>"
#  .} = object
#
#  NeighborRange {.
#    importcpp: "NetworKit::Graph::NeighborRange", header: #"networkit/graph/Graph.hpp"
#  .} = object
#
#  NeighborIterator {.
#    importcpp: "NetworKit::Graph::NeighborIterator",
#    header: "<networkit/graph/Graph.hpp>"
#  .} = object
#
#  NodeVector {.importcpp: "std::vector<NetworKit::node>", #header: "<vector>".} = object
#
## forward declare functions to prevent circular dependencies
#proc getBetweenness*(g: Graph): Table[int, float]
#proc getCloseness*(g: Graph): Table[int, float]
#proc getRandomNode*(g: Graph): int
#proc getAttributeAssortativity*(s: State): Table[int, float]
#proc getDegreeCentrality*(g: Graph): Table[int, float]
#
#proc newNodeVector(): NodeVector {.
#  importcpp: "std::vector<NetworKit::node>()", constructor
#.}
#
#proc add(v: var NodeVector, node: int) {.importcpp: "#.push_back#(@)".}
#proc len(v: NodeVector): int {.importcpp: "#.size()".}
#proc `[]`(v: NodeVector, i: int): int {.importcpp: "#[#]".}
#
#proc newGraph*(nodes: int): Graph {.importcpp: "NetworKit::Graph#(@)", constructor.}
#proc addEdge*(g: var Graph, u, v: int) {.importcpp: "#.addEdge#(@, @)".}
#proc numberOfNodes*(g: Graph): int {.importcpp: "#.numberOfNodes(#)".}
#
#proc numberOfEdges*(g: Graph): int {.importcpp: "#.numberOfEdges(#)".}
#
#proc u*(e: Edge): int {.importcpp: "#.u".}
#proc v*(e: Edge): int {.importcpp: "#.v".}
#
#proc newBetweenness(
#  g: Graph
#): Betweenness {.importcpp: "NetworKit::Betweenness(@)", #constructor.}
#
#proc newCloseness(
#  g: Graph, normalized: bool, variant: ClosenessVariant
#): Closeness {.importcpp: "NetworKit::Closeness(@)", constructor.#}
#
#proc newDegreeCentrality(
#  g: Graph
#): DegreeCentrality {.importcpp: "NetworKit::DegreeCentrality(@)#", constructor.}
#
#proc run(b: var Betweenness) {.importcpp: "#.run()".}
#proc run(c: var Closeness) {.importcpp: "#.run()".}
#proc run(d: var DegreeCentrality) {.importcpp: "#.run()".}
#
#proc scores(b: Betweenness): Vector {.importcpp: "#.scores()".}
#proc scores(c: Closeness): Vector {.importcpp: "#.scores()".}
#proc scores(d: DegreeCentrality): Vector {.importcpp: "#.scores()#".}
#
#proc newNodeSet*(): NodeSet {.
#  importcpp: "std::unordered_set<NetworKit::node>()", constructor
#.}
#
#proc insert*(s: var NodeSet, node: int) {.importcpp: "#.insert(@)#".}
#
#proc edgeRange*(g: Graph): EdgeRange {.importcpp: "#.edgeRange()#".}
#
#proc nodeRange(g: Graph): NodeRange {.importcpp: "#.nodeRange()".#}
#proc begin(it: NodeRange): NodeIterator {.importcpp: "#.begin()".#}
#proc `end`(it: NodeRange): NodeIterator {.importcpp: "#.end()".}
#proc `!=`*(a, b: NodeIterator): bool {.importcpp: "(# != #)".}
#proc `*`*(it: NodeIterator): int {.importcpp: "*#".}
#proc inc*(it: var NodeIterator) {.importcpp: "++#".}
#
#iterator nodes(g: Graph): int =
#  let nodeRange = g.nodeRange()
#  var it = nodeRange.begin()
#  let endIt = nodeRange.end()
#  while it != endIt:
#    yield `*`(it)
#    it.inc
#
#proc randomNode(
#  g: Graph
#): int {.importcpp: "NetworKit::GraphTools::randomNodes(@, 1)[0]#".}
#
#proc begin(it: EdgeIterator): EdgeIterator {.importcpp: "#.begin(#)".}
#
#proc `end`(it: EdgeIterator): EdgeIterator {.importcpp: "#.end()#".}
#proc `!=`*(a, b: EdgeIterator): bool {.importcpp: "(# != #)".}
#proc `*`*(it: EdgeIterator): Edge {.importcpp: "*#".}
#proc inc*(it: var EdgeIterator) {.importcpp: "++#".}
#iterator items(n: EdgeIterator): int =
#  var it = n.begin()
#  let endIt = n.`end`()
#  while it != endIt:
#    yield (`*`(it)).v
#    it.inc
#
#proc begin*(range: EdgeRange): EdgeIterator {.importcpp: "#.begin#()".}
#proc `end`*(range: EdgeRange): EdgeIterator {.importcpp: "#.end()#".}
#iterator edges*(g: Graph): Edge =
#  let edges = g.edgeRange()
#  var it = edges.begin()
#  let endIt = edges.end()
#  var idx = 0
#  while it != endIt:
#    let edge = `*` it
#    yield edge
#    it.inc
#
#proc neighborRange*(
#  g: Graph, node: int
#): NeighborRange {.importcpp: "#.neighborRange(@)".}
#
#proc begin*(range: NeighborRange): NeighborIterator {.importcpp: #"#.begin()".}
#proc `end`*(range: NeighborRange): NeighborIterator {.importcpp: #"#.end()".}
#
#proc `!=`*(a, b: NeighborIterator): bool {.importcpp: "(# != #)".#}
#proc `*`*(it: NeighborIterator): int {.importcpp: "*#".}
#proc inc*(it: var NeighborIterator) {.importcpp: "++#".}
#
#iterator neighbors*(g: Graph, node: int): int =
#  let neighborRange = g.neighborRange(node)
#  var it = neighborRange.begin()
#  let endIt = neighborRange.`end`()
#  while it != endIt:
#    yield `*` it
#    inc(it)
#
#proc removeEdge*(g: var Graph, i, j: int) {.importcpp: #"#.removeEdge(#, #)".}
#
#proc addNode*(g: var Graph, node: int) {.importcpp: "#.addNode(@)#".}
#proc removeNode*(g: var Graph, node: int) {.importcpp: #"#.removeNode(@)".}
#proc hasNode*(g: Graph, node: int): bool {.importcpp: "#.hasNode#(@)".}
#proc upperNodeIdBound*(g: Graph): int {.importcpp: #"#.upperNodeIdBound()".}
#
#proc `[]`(v: Vector, i: int): float {.importcpp: "#[#]".}
#
#proc `[]`(m: NodeMap, key: int): float {.importcpp: "#[#]".}
#proc `[]=`(m: var NodeMap, key: int, val: float) {.importcpp: "##[#] = #".}
#
#proc subgraphFromNodes*(
#  G: Graph, nodes: openArray[int], compact: bool = false
#): Graph {.
#  importcpp:
#    """
#  [&](const auto& G, auto nodes, bool compact) {
#    return NetworKit::GraphTools::subgraphFromNodes(G, #nodes.begin(), nodes.end(), compact);
#  }(#, #, #)
#"""
#.}
#
## Alternative version using HashSet
#proc subgraphFromNodesSet*(
#  G: Graph, nodes: NodeSet, compact: bool = false
#): Graph {.
#  importcpp:
#    """
#  [&](const auto& G, const auto& nodes, bool compact) {
#    return NetworKit::GraphTools::subgraphFromNodes(G, #nodes.begin(), nodes.end(), compact);
#  }(#, #, #)
#""",
#  header: "<networkit/graph/GraphTools.hpp>"
#.}
#
#proc getEgoNetwork*(g: Graph, ego: int): Graph =
#  var egoNetworkNodes = newNodeSet()
#  egoNetworkNodes.insert(ego)
#
#  for neighbor in g.neighbors(ego):
#    egoNetworkNodes.insert(neighbor)
#
#  result = subgraphFromNodesSet(g, egoNetworkNodes)
#
#type
#  NodeIntAttribute* {.
#    importcpp: "NetworKit::Graph::NodeIntAttribute",
#    header: "<networkit/graph/Graph.hpp>"
#  .} = object
#
#  NodeDoubleAttribute* {.
#    importcpp: "NetworKit::Graph::NodeDoubleAttribute",
#    header: "<networkit/graph/Graph.hpp>"
#  .} = object
#
#  NodeStringAttribute* {.
#    importcpp: "NetworKit::Graph::NodeStringAttribute",
#    header: "<networkit/graph/Graph.hpp>"
#  .} = object
#
#  StdString {.importcpp: "std::string", header: "<string>".} = #object
#
#proc newStdString(s: cstring): StdString {.importcpp: #"std::string(@)", constructor.}
#proc `$`(s: StdString): string {.importcpp: "(#).c_str()".}
#
## Attribute attachment
#proc attachNodeIntAttribute*(
#  g: var Graph, name: StdString
#): NodeIntAttribute {.importcpp: "#.attachNodeIntAttribute(@)".}
#
#proc attachNodeDoubleAttribute*(
#  g: var Graph, name: StdString
#): NodeDoubleAttribute {.importcpp: "#.attachNodeDoubleAttribute#(@)".}
#
#proc attachNodeStringAttribute*(
#  g: var Graph, name: StdString
#): NodeStringAttribute {.importcpp: "#.attachNodeStringAttribute#(@)".}
#
## Attribute detachment
#proc detachNodeAttribute*(
#  g: var Graph, name: StdString
#) {.importcpp: "#.detachNodeAttribute(@)".}
#
## Helper procedures for setting and getting attribute values
#proc `[]=`*(
#  attr: var NodeIntAttribute, node: int, value: int
#) {.importcpp: "#[#] = std::move(#)".}
#
#proc `[]=`*(
#  attr: var NodeDoubleAttribute, node: int, value: float64
#) {.importcpp: "#[#] = std::move(#)".}
#
#proc `[]=`*(
#  attr: var NodeStringAttribute, node: int, value: StdString
#) {.importcpp: "#[#] = std::move(#)".}
#
#proc `[]`*(attr: NodeIntAttribute, node: int): int {.importcpp: #"#[#]".}
#proc `[]`*(attr: NodeDoubleAttribute, node: int): float64 #{.importcpp: "#[#]".}
#proc `[]`*(attr: NodeStringAttribute, node: int): StdString #{.importcpp: "#[#]".}
## Helper function to get all node attributes as a table
#
#proc toGraph*(s: State): Graph =
#  echo "Creating graph with ", s.agents.len, " nodes"
#  result = newGraph(s.agents.len)
#
#  echo "Adding edges"
#  for (i, j) in s.toAdj():
#    result.addEdge(i, j)
#  return result
#
#  echo "Attaching role attribute"
#  var roleAttr: NodeStringAttribute
#  try:
#    roleAttr = result.attachNodeStringAttribute(newStdString(#"role"))
#  except:
#    echo "Exception when attaching role attribute: ", #getCurrentExceptionMsg()
#    return result
#
#  echo "Attaching state attribute"
#  var stateAttr: NodeDoubleAttribute
#  try:
#    stateAttr = result.attachNodeDoubleAttribute(newStdString(#"state"))
#  except:
#    echo "Exception when attaching state attribute: ", #getCurrentExceptionMsg()
#    return result
#
#  echo "Setting attributes for agents"
#  for agent in s.agents:
#    echo fmt"Setting attributes for agent {agent.id}"
#    try:
#      agent.role = "A"
#      echo agent.id, " ", agent.role, " ", agent.state
#      let r = newStdString(agent.role)
#
#      roleAttr[agent.id] = r
#      stateAttr[agent.id] = agent.state.float64
#    except:
#      echo fmt"Exception when setting attributes for agent #{agent.id}: ",
#        getCurrentExceptionMsg()
#
#  echo "Graph creation complete"
#
#proc getNodeAttributes*[T](g: Graph, name: string): Table[int, T#] =
#  result = initTable[int, T]()
#  when T is int:
#    var attr = g.attachNodeIntAttribute(name.cstring)
#  elif T is float:
#    var attr = g.attachNodeDoubleAttribute(name.cstring)
#  elif T is string:
#    var attr = g.attachNodeStringAttribute(name.cstring)
#  else:
#    {.error: "Unsupported attribute type".}
#
#  for node in 0 ..< g.numberOfNodes():
#    when T is string:
#      result[node] = $attr[node]
#    else:
#      result[node] = T(attr[node])
#
#  # Detach the attribute after we're done
#  g.detachNodeAttribute(name.cstring)
#
#proc getBetweenness*(g: Graph): Table[int, float] =
#  var betweenness = newBetweenness(g)
#  betweenness.run()
#  let betweennessScores = betweenness.scores()
#  result = initTable[int, float]()
#  for node in (0 ..< g.numberOfNodes()):
#    result[node] = betweennessScores[node].float
#
#proc getCloseness*(g: Graph): Table[int, float] =
#  var closeness = newCloseness(g, true, ClosenessVariant.Standard#)
#  closeness.run()
#  let closenessScores = closeness.scores()
#  result = initTable[int, float]()
#  for node in (0 ..< g.numberOfNodes()):
#    result[node] = closenessScores[node].float
#
#proc getDegreeCentrality*(g: Graph): Table[int, float] =
#  var degreeCentrality = newDegreeCentrality(g)
#  degreeCentrality.run()
#  let degreeCentralityScores = degreeCentrality.scores()
#  result = initTable[int, float]()
#  for node in (0 ..< g.numberOfNodes()):
#    result[node] = degreeCentralityScores[node].float
#
#proc getAttributeAssortativity*(s: State): Table[int, float] =
#  var g = s.toGraph
#  var roleMap = initTable[string, float]()
#  for idx, role in s.config.p_roles.keys.toSeq():
#    roleMap[role] = idx.float
#
#  var attributes = newSeq[float](s.agents.len)
#  for agent in s.agents:
#    attributes[agent.id] = roleMap[agent.role]
#
#  proc catAssortativity(g: Graph, center: int): float =
#    let ego = g.getEgoNetwork(center)
#    let m = ego.numberOfEdges().float
#    if m == 0:
#      return 0.0 # No edges in ego network
#
#    var ai, bi: Table[int, float64]
#    var eii = 0.0
#    for edge in ego.edges():
#      let x = attributes[edge.u.int].int
#      let y = attributes[edge.v.int].int
#      if x == y:
#        eii += 1.0
#      ai.mgetOrPut(x, 0.0) += 0.5
#      ai.mgetOrPut(y, 0.0) += 0.5
#      bi.mgetOrPut(x, 0.0) += 0.5
#      bi.mgetOrPut(y, 0.0) += 0.5
#
#    eii /= m
#    var a = 0.0
#    for k, v in ai:
#      a += v * bi[k]
#    a /= m * m
#
#    if a == 1.0:
#      return 1.0 # Perfect assortativity
#
#    result = (eii - a) / (1.0 - a)
#
#  result = initTable[int, float]()
#  for node in 0 ..< g.numberOfNodes():
#    result[node] = catAssortativity(g, node)
#
#proc getRandomNode*(g: Graph): int =
#  if g.numberOfNodes() == 0:
#    return 0
#  return g.randomNode().int
