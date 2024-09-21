type Intervention* = tuple[which, kind: string, n: int, onlyCriminals: bool]
from boiler_room/agent import State, hasNeighbor, addEdge, rmEdge

from boiler_room/graph import
  Graph, getBetweenness, getDegreeCentrality, getAttributeAssortativity, getCloseness,
  getRandomNode, newNodeSet, toGraph, insert, subgraphFromNodesSet

import math, sequtils, tables, strformat, random

proc toGraph*(s: State, intervention: Intervention): Graph =
  result = s.toGraph()
  # create a subgraph only with the criminals
  if intervention.onlyCriminals:
    var criminals = newNodeSet()
    for agent in s.agents:
      if agent.state == 1.0:
        criminals.insert(agent.id)
    result = result.subgraphFromNodesSet(criminals)

proc getMax(s: State, intervention: Intervention): int {.gcsafe, thread.} =
  var options = initTable[int, float]()
  let g = s.toGraph(intervention)
  case intervention.kind
  of "degree":
    options = getDegreeCentrality(g)
  of "role":
    options = getAttributeAssortativity(s)
  of "betweenness":
    options = getBetweenness(g)
  of "closeness":
    options = getCloseness(g)
  of "random":
    let node = getRandomNode(g)
    options[node] = 1.0
  else:
    raise newException(
      ValueError, &"Intervention = {intervention.kind} not understood check source"
    )

  let tmp = options.values().toseq()
  if tmp.len == 0:
    return 0
  var t: float
  if intervention.kind != "role":
    t = max(tmp)
  # target the most dissasortative node
  else:
    t = min(tmp)
  for key, value in options:
    if value == t:
      return key

proc addRandomEdge(s: var State, focal: int) =
  var agents = (0 ..< s.agents.len).toSeq()
  s.rng.shuffle(agents)
  for other in agents:
    if other != focal and not s.agents[focal].hasNeighbor(other):
      s.agents[focal].addEdge s.agents[other]
      return

proc intervene*(state: var State, intervention: Intervention) {.gcsafe, thread.} =
  # Convert the state to a networkx graph
  if intervention.kind == "none":
    return

  # Get the max centrality value
  let target = getMax(state, intervention)
  if intervention.kind == "remove":
    # Remove the edges of the target
    let neighbors = state.agents[target].neighbors.keys().toSeq()
    for neighbor in neighbors:
      state.agents[target].rmEdge state.agents[neighbor]
  elif intervention.kind == "add":
    state.addRandomEdge(target)
  elif intervention.kind == "state":
    state.agents[target].state = 0.0
  else:
    raise newException(
      ValueError, &"Intervention={intervention.kind} not understood check source"
    )
