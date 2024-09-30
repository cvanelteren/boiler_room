import sequtils, sets, tables, math
from boiler_room.agent import State, Agent

proc degreeCentrality*(state: State): Table[int, float] {.inline.} =
  result = initTable[int, float]()
  let n = state.agents.len.float
  for agent in state.agents:
    result[agent.id] = agent.neighbors.len.float

proc closenessCentrality*(state: State): Table[int, float] {.inline.} =
  result = initTable[int, float]()
  let n = state.agents.len

  # Create a mapping between original IDs and sequential indices
  var idToIndex = initTable[int, int]()
  var indexToId = newSeq[int](n)
  for i, agent in state.agents:
    idToIndex[agent.id] = i
    indexToId[i] = agent.id

  proc bfs(start: int): seq[int] =
    var distances = newSeq[int](n)
    var queue = @[start]
    distances[start] = 0
    var index = 0
    while index < queue.len:
      let current = queue[index]
      let currentAgent = state.agents[current]
      for neighborId in currentAgent.neighbors.keys:
        let neighbor = idToIndex[neighborId]
        if distances[neighbor] == 0 and neighbor != start:
          distances[neighbor] = distances[current] + 1
          queue.add(neighbor)
      index.inc
    return distances

  for i, agent in state.agents:
    let distances = bfs(i)
    let sum_distances = distances.filterIt(it > 0).sum()
    if sum_distances > 0:
      result[agent.id] = (n - 1).float / sum_distances.float
    else:
      result[agent.id] = 0.0

  return result

proc betweennessCentrality*(state: State): Table[int, float] {.inline.} =
  result = initTable[int, float]()
  let n = state.agents.len

  # Create a mapping between original IDs and sequential indices
  var idToIndex = initTable[int, int]()
  var indexToId = newSeq[int](n)
  for i, agent in state.agents:
    idToIndex[agent.id] = i
    indexToId[i] = agent.id
    result[agent.id] = 0.0

  proc bfs(start: int): (seq[float], seq[seq[int]], seq[int]) =
    var sigma = newSeq[float](n)
    var distances = newSeq[int](n)
    var predecessors = newSeq[seq[int]](n)
    var queue = @[start]
    var stack: seq[int] = @[]
    sigma[start] = 1.0
    distances[start] = 0
    var index = 0
    while index < queue.len:
      let current = queue[index]
      stack.add(current)
      let currentAgent = state.agents[current]
      for neighborId in currentAgent.neighbors.keys:
        let neighbor = idToIndex[neighborId]
        if distances[neighbor] == 0 and neighbor != start:
          queue.add(neighbor)
          distances[neighbor] = distances[current] + 1
        if distances[neighbor] == distances[current] + 1:
          sigma[neighbor] += sigma[current]
          predecessors[neighbor].add(current)
      index.inc
    return (sigma, predecessors, stack)

  for s in 0 ..< n:
    var (sigma, predecessors, stack) = bfs(s)
    var delta = newSeq[float](n)
    while stack.len > 0:
      let w = stack.pop()
      for v in predecessors[w]:
        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
      if w != s:
        result[indexToId[w]] += delta[w]

  # Normalize
  let normFactor = (n - 1) * (n - 2)
  for v in result.keys:
    if normFactor > 0:
      result[v] /= normFactor.float
    else:
      result[v] = 0.0

  return result

proc normalizedCentrality*(centrality: Table[int, float]): Table[int, float] =
  result = initTable[int, float]()
  let maxValue = max(toSeq(centrality.values))
  for k, v in centrality.pairs:
    result[k] = v / maxValue

proc subgraphFrom*(graph: State, nodeIds: seq[int]): State {.inline.} =
  var subgraphAgents: seq[Agent] = @[]
  let nodeIdSet = nodeIds.toHashSet()

  # Create a lookup table for faster agent retrieval
  var agentLookup = initTable[int, Agent]()
  for agent in graph.agents:
    agentLookup[agent.id] = agent

  # Create new agents for the subgraph
  for id in nodeIds:
    if id in agentLookup:
      let originalAgent = agentLookup[id]
      var newNeighbors = initTable[int, int]()

      # Only include neighbors that are also in the subgraph
      for neighborId, weight in originalAgent.neighbors:
        if neighborId in nodeIdSet:
          newNeighbors[neighborId] = weight

      let newAgent =
        Agent(id: originalAgent.id, role: originalAgent.role, neighbors: newNeighbors)
      subgraphAgents.add(newAgent)

  return State(agents: subgraphAgents)

proc createEgoGraph*(state: State, agentId: int): State {.inline.} =
  # Create a set of agent IDs to include in the ego graph
  var egoNodeIds = initHashSet[int]()

  # Find the focal agent
  let focalAgent = state.agents.filterIt(it.id == agentId)
  if focalAgent.len == 0:
    raise newException(ValueError, "Agent with ID " & $agentId & " not found")

  egoNodeIds.incl(agentId)
  for neighborId in focalAgent[0].neighbors.keys:
    egoNodeIds.incl(neighborId)

  # Create a lookup table for faster agent retrieval
  var agentLookup = initTable[int, Agent]()
  for agent in state.agents:
    agentLookup[agent.id] = agent

  # Create new Agent objects for the ego graph
  var egoAgents: seq[Agent] = @[]
  for id in egoNodeIds:
    if id notin agentLookup:
      echo "Warning: Agent with ID ", id, " not found in the original graph"
      continue

    let originalAgent = agentLookup[id]
    var newNeighbors = initTable[int, int]()
    for neighborId, weight in originalAgent.neighbors:
      if neighborId in egoNodeIds and neighborId in agentLookup:
        newNeighbors[neighborId] = weight

    let newAgent = Agent(
      id: originalAgent.id,
      state: originalAgent.state,
      role: originalAgent.role,
      neighbors: newNeighbors,
      bias: originalAgent.bias,
      nSamples: originalAgent.nSamples,
      edgeRate: originalAgent.edgeRate,
      mutationRate: originalAgent.mutationRate,
    )
    egoAgents.add(newAgent)

  # Create and return the new State object representing the ego graph
  result =
    State(agents: egoAgents, valueNetwork: state.valueNetwork, config: state.config)

proc roleAssortativity*(graph: State, focalAgentId: int): float {.inline.} =
  var roleCounts = initCountTable[string]()
  var edgesByRoles = initTable[string, CountTable[string]]()
  var totalEdges = 0
  let focalAgent = graph.agents.filterIt(it.id == focalAgentId)[0]

  # Count roles and initialize edgesByRoles
  for agent in graph.agents:
    roleCounts.inc(agent.role)
    if agent.role notin edgesByRoles:
      edgesByRoles[agent.role] = initCountTable[string]()

  # Count edges between roles (undirected), focusing on focal agent's connections
  for neighborId in focalAgent.neighbors.keys:
    let neighbor = graph.agents.filterIt(it.id == neighborId)[0]
    edgesByRoles[focalAgent.role].inc(neighbor.role)
    edgesByRoles[neighbor.role].inc(focalAgent.role)
    totalEdges += 2 # Count each edge twice for undirected graph

  # Calculate assortativity
  var sum_e = 0.0
  var sum_a = 0.0

  for role in roleCounts.keys:
    let e_ii = edgesByRoles[role][role].float / totalEdges.float
    let a_i = edgesByRoles[role].values.toSeq.sum.float / totalEdges.float
    sum_e += e_ii
    sum_a += a_i * a_i

  result =
    if sum_a != 1:
      (sum_e - sum_a) / (1 - sum_a)
    else:
      1.0

  # Handle edge cases
  if totalEdges == 0: # No edges in the ego network
    result = 0.0

proc roleAssortativityCentrality*(state: State): Table[int, float] {.inline.} =
  result = initTable[int, float]()

  for agent in state.agents:
    let egoGraph = createEgoGraph(state, agent.id)
    result[agent.id] = roleAssortativity(egoGraph, agent.id)
