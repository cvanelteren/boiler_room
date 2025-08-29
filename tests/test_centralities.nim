import unittest, tables, sequtils, math
import boiler_room/[agent, centralities]

# Helper function to create a test network
proc createTestNetwork(): State =
  #   1  - 2
  #    \   / \
  #      0  -  3

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
    p_roles: {"A": 0.15, "B": 0.7, "C": 0.15}.toTable,
  )

  result = State(
    agents: @[],
    valueNetwork: {
      "A": {"B": 1.0, "C": 1.0}.toTable,
      "B": {"A": 1.0, "C": 1.0}.toTable,
      "C": {"A": 1.0, "B": 1.0}.toTable,
    }.toTable,
    config: mockConfig,
  )
  for idx in (0 .. 2):
    discard makeAgent(idx, result)
  result.agents[0].addEdge(result.agents[1])
  result.agents[1].addEdge(result.agents[2])
  result.agents[0].role = "A"
  result.agents[1].role = "B"
  result.agents[2].role = "C"

suite "Centrality Measures":
  let testNetwork = createTestNetwork()

  test "Degree Centrality":
    let degree = degreeCentrality(testNetwork)
    require:
      degree[0] == 1.0
      degree[1] == 2.0
      degree[2] == 1.0

  test "Closeness Centrality":
    let closeness = closenessCentrality(testNetwork)
    require:
      abs(closeness[0] - 2 / 3) < 1e-6
      abs(closeness[1] - 1.0) < 1e-6
      abs(closeness[2] - 2 / 3) < 1e-6

  test "Betweenness Centrality":
    let betweenness = betweennessCentrality(testNetwork)
    require:
      abs(betweenness[0]) < 1e-6
      abs(betweenness[1] - 1.0) < 1e-6
      abs(betweenness[2]) < 1e-6

  test "Role Assortativity":
    let roleAssortativity = roleAssortativityCentrality(testNetwork)
    echo roleAssortativity
    require:
      roleAssortativity[0] == -1.0
      abs(roleAssortativity[1] + 0.6) < 1e-6
      roleAssortativity[2] == -1.0
  test "Role Assortativity in Star Graph":
    let starGraph = State(
      agents:
        @[
          Agent(id: 0, role: "A", neighbors: initTable[int, int]()),
          Agent(id: 1, role: "B", neighbors: initTable[int, int]()),
          Agent(id: 2, role: "B", neighbors: initTable[int, int]()),
          Agent(id: 3, role: "B", neighbors: initTable[int, int]()),
          Agent(id: 4, role: "B", neighbors: initTable[int, int]()),
        ]
    )
    starGraph.agents[0].addEdge(starGraph.agents[1])
    starGraph.agents[0].addEdge(starGraph.agents[2])
    starGraph.agents[0].addEdge(starGraph.agents[3])
    starGraph.agents[0].addEdge(starGraph.agents[4])

    let roleAssortativity = roleAssortativityCentrality(starGraph)
    echo roleAssortativity
    require:
      (roleAssortativity[0] + 1.0).abs < 1e-6
      (roleAssortativity[1] + 1.0).abs < 1e-6
      (roleAssortativity[2] + 1.0).abs < 1e-6
      (roleAssortativity[3] + 1.0).abs < 1e-6
      (roleAssortativity[4] + 1.0).abs < 1e-6

  test "Edge Cases":
    var singleNodeNetwork =
      State(agents: @[Agent(id: 0, role: "A", neighbors: initTable[int, int]())])
    let degreeS = degreeCentrality(singleNodeNetwork)
    let closenessS = closenessCentrality(singleNodeNetwork)
    let betweennessS = betweennessCentrality(singleNodeNetwork)
    let roleAssortS = roleAssortativityCentrality(singleNodeNetwork)

    require(degreeS[0] == 0.0)
    require(closenessS[0] == 0.0)
    require(betweennessS[0] == 0.0)
    require(roleAssortS[0] == 0.0)

suite "Centrality Measure Properties":
  let testNetwork = createTestNetwork()

  test "Degree Centrality Range":
    let degree = degreeCentrality(testNetwork)
    let z = testNetwork.agents.len - 1
    for v in degree.values:
      check(v >= 0.0 and v <= z.float)

  test "Closeness Centrality Range":
    let closeness = closenessCentrality(testNetwork)
    for v in closeness.values:
      check(v >= 0.0 and v <= 1.0)

  test "Betweenness Centrality Non-Negative":
    let betweenness = betweennessCentrality(testNetwork)
    for v in betweenness.values:
      check(v >= 0.0)

  test "Role Assortativity Range":
    let roleAssortativity = roleAssortativityCentrality(testNetwork)
    for v in roleAssortativity.values:
      check(v >= -1.0 and v <= 1.0)
