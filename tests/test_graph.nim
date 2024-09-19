import unittest
import sequtils, random, tables, math, sets
import nimpy
import boiler_room
import boiler_room.graph

{.passC: "-std=gnu++17".}
{.passL: "-lstdc++".}

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
  getBetweenness, getCloseness, getDegreeCentrality, getAttributeAssorativity

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
    let state = createMockState(10)
    let graph = state.toGraph
    let egoNetwork = graph.getEgoNetwork(0)
    check(egoNetwork.numberOfNodes() > 0)
    check(egoNetwork.numberOfNodes() <= graph.numberOfNodes())

suite "Centrality Measures":
  test "Betweenness Centrality":
    let state = createMockState(10)
    let betweenness = state.getBetweenness()
    check(betweenness.len == 10)
    for _, value in betweenness:
      check(value >= 0.0)

  test "Closeness Centrality":
    let state = createMockState(10)
    let closeness = state.getCloseness()
    check(closeness.len == 10)
    for _, value in closeness:
      check(value >= 0.0 and value <= 1.0)

  test "Degree Centrality":
    let state = createMockState(10)
    let degree = state.getDegreeCentrality()
    check(degree.len == 10)
    for _, value in degree:
      check(value >= 0.0)

suite "Attribute Assortativity":
  test "Attribute Assortativity Calculation":
    let state = createMockState(20)
    #let assortativity = state.getAttributeAssorativity()
    #for node, assortativity in assortativity:
    #check(assortativity >= -1.0 and assortativity <= 1.0)

  test "Attribute Assortativity for Homogeneous Network":
    var state = createMockState(10)
    for agent in state.agents.mitems:
      agent.role = "A"
    #let assortativities = state.getAttributeAssorativity()
    #for node, assortativity in assortativities:
    #check(assortativity.abs < 1e-6) # Should be close to 0 for homogeneous network

  test "Attribute Assortativity for Perfectly Assortative Network":
    var state = createMockState(10)
    for i, agent in state.agents.mpairs:
      agent.role = if i < 5: "A" else: "B"
    for i in 0 ..< 5:
      for j in 1 .. 4:
        state.agents[i].addEdge(state.agents[i + j]) # Connect only to different role
    #let assortativities = state.getAttributeAssorativity()
    #for node, assortativity in assortativities:
    #  check(assortativity > 0.9) # Should be close to 1 for perfectly assortative network

suite "Graph Modifications":
  test "Remove Node":
    var state = createMockState(5)
    var graph = state.toGraph
    let initialNodes = graph.numberOfNodes()
    graph.removeNode(2)
    check(graph.numberOfNodes() == initialNodes - 1)
    check(not graph.hasNode(2))

  test "Adding and removing edges":
    var state = createMockState(5)
    var graph = state.toGraph
    for idx in 0 ..< 5:
      graph.addEdge(idx, (idx + 1) mod 5)
    check graph.numberOfEdges() == 6
    graph.removeEdge(0, 1)
    check graph.numberOfEdges() == 5
