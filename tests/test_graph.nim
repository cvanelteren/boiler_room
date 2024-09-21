import unittest
import sequtils, random, tables, math, sets
import nimpy
import boiler_room
import boiler_room.graph

{.passC: "-std=gnu++17".}
{.passL: "-lstdc++".}

# TODO: would be nice to make this dynamic or passed on as when built
{.
  passC: "-I /home/casper/micromamba/envs/boiler/lib/python3.12/site-packages/include/"
.}
{.
  passL:
    "-L/home/casper/micromamba/envs/boiler/lib/python3.12/site-packages -l:libnetworkit.so -Wl,-rpath,/home/casper/micromamba/envs/boiler/lib/python3.12/site-packages"
.}

from boiler_room.graph import
  Graph, numberOfNodes, hasNode, addNode, removeNode, addEdge, removeEdge,
  getEgoNetwork, toGraph
from boiler_room.graph import
  getBetweenness, getCloseness, getDegreeCentrality, getAttributeAssortativity

randomize(42) # Set a fixed seed for reproducibility

let mockConfig = Config(
  beta: 0.1,
  benefit: 1.0,
  cost: 0.5,
  depth: 1,
  n_samples: 10,
  t: 100,
  seed: 42,
  z: 5,
  trial: 1,
  step: 1,
  p_states: {0.0: 0.5, 1.0: 0.5}.toTable,
  p_roles: {"A": 0.3, "B": 0.7}.toTable,
)

# Helper function to create a mock State object
proc createMockState(numAgents: int): State =
  result = State(
    agents: @[],
    rng: initRand(42),
    valueNetwork:
      {"A": {"A": 1.0, "B": 0.5}.toTable, "B": {"A": 0.5, "B": 1.0}.toTable}.toTable,
    config: mockConfig,
  )
  for i in 0 ..< numAgents:
    var a = makeAgent(i, result)
    a.state = if i mod 2 == 0: 0.0 else: 1.0
    a.role = if i mod 3 == 0: "A" else: "B"
    a.edgeRate = 0.1
    a.mutationRate = 0.01

  # Add some edges
  for i in 0 ..< numAgents:
    for j in 0 ..< numAgents:
      if i != j and rand(1.0) < 0.2: # 20% chance of edge
        result.agents[i].addEdge(result.agents[j])

suite "Graph Operations":
  test "State to Graph conversion":
    let state = createMockState(5)
    let graph = state.toGraph
    check(graph.numberOfNodes() == 5)
    # You might want to add more specific checks for edges

  test "Ego Network":
    var g = newGraph(5)
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(0, 3)
    g.addEdge(0, 4)
    let egoNetwork = g.getEgoNetwork(ego = 0)
    check egoNetwork.numberOfNodes() == 5
    echo egoNetwork.numberOfEdges() == 4

suite "Graph Modifications":
  test "Remove Node":
    var state = createMockState(5)
    var graph = state.toGraph
    let initialNodes = graph.numberOfNodes()
    graph.removeNode(2)
    check(graph.numberOfNodes() == initialNodes - 1)
    check(not graph.hasNode(2))

  test "Adding and removing edges":
    var graph = newGraph(5)
    for idx in 0 ..< 5:
      graph.addEdge(idx, (idx + 1) mod 5)
    check graph.numberOfEdges() == 5
    graph.removeEdge(0, 1)
    check graph.numberOfEdges() == 4

suite "Centrality Measures":
  test "Betweenness Centrality":
    let state = createMockState(10)
    let g = state.toGraph()
    let betweenness = g.getBetweenness()
    check(betweenness.len == 10)
    for _, value in betweenness:
      check(value >= 0.0)

  test "Closeness Centrality":
    let state = createMockState(10)
    let g = state.toGraph()
    let closeness = g.getCloseness()
    check(closeness.len == 10)
    for _, value in closeness:
      check(value >= 0.0 and value <= 1.0)

  test "Degree Centrality":
    let state = createMockState(10)
    let g = state.toGraph()
    let degree = g.getDegreeCentrality()
    check(degree.len == 10)
    for _, value in degree:
      check(value >= 0.0)

suite "Attribute Assortativity":
  test "Attribute Assortativity Calculation":
    let state = createMockState(20)
    let assortativity = state.getAttributeAssortativity()
    for node, assortativity in assortativity:
      require(assortativity >= -1.0 and assortativity <= 1.0)

  test "Attribute Assortativity for Perfectly Assortative Network":
    var state = createMockState(2)
    state.agents[0].role = "A"
    state.agents[1].role = "A"
    state.agents[0].addEdge(state.agents[1])
    let assortativities = state.getAttributeAssortativity()
    for node, assortativity in assortativities:
      # Should be close to 1 for perfectly assortative network
      require assortativity == 1.0
  test "Perfect dissorativity":
    var state = createMockState(2)
    state.agents[0].role = "A"
    state.agents[1].role = "B"
    state.agents[0].addEdge(state.agents[1])
    let assortativities = state.getAttributeAssortativity()
    for node, assortativity in assortativities:
      # Should be close to -1 for perfectly disassortative network
      require assortativity == -1.0
  test "Zero Assortativity":
    var state = createMockState(3)
    state.agents[0].role = "A"
    state.agents[1].role = "B"
    state.agents[2].role = "A"
    for agent in state.agents:
      agent.neighbors.clear()
      require agent.neighbors.len == 0
    state.agents[0].addEdge(state.agents[1])
    state.agents[0].addEdge(state.agents[2])

    let assortativities = state.getAttributeAssortativity()
    for node, assortativity in assortativities:
      if node == 0:
        require (assortativity + 0.3333333333) < 1e-6
      elif node == 1:
        require assortativity == -1.0
      elif node == 2:
        require assortativity == 1.0
