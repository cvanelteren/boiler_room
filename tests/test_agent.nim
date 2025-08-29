import unittest
import sequtils, random, tables, math, sets
import nimpy
import boiler_room
#import boiler_room.agent # Assuming the code you provided is in a file named agent.nim

# Mock PyObject for testing
type MockPyObject = ref object

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
suite "Agent Module Tests":
  test "makeAgent creates an Agent with correct properties":
    var state = State(config: mockConfig, rng: initRand(42))
    let agent = makeAgent(0, state)
    let other = makeAgent(1, state)

    require:
      agent.id == 0
      agent.neighbors.len == 0
      agent.role in ["A", "B"]
      agent.state in [0.0, 1.0]
      agent.nSamples == mockConfig.n_samples
      agent.parent == state
      agent.state == other.state
      agent == state.agents[0]
      other == state.agents[1]

    expect ValueError:
      discard makeAgent(0, state)

  test "addEdge adds edge correctly":
    var agent1 = Agent(id: 1, neighbors: initTable[int, int]())
    var agent2 = Agent(id: 2, neighbors: initTable[int, int]())

    agent1.addEdge(agent2)

    require:
      agent1.neighbors.hasKey(2)
      agent2.neighbors.hasKey(1)

  test "rmEdge removes edge correctly":
    var agent1 = Agent(id: 1, neighbors: {2: 1}.toTable)
    var agent2 = Agent(id: 2, neighbors: {1: 1}.toTable)

    agent1.rmEdge(agent2)

    require:
      not agent1.neighbors.hasKey(2)
      not agent2.neighbors.hasKey(1)

  test "fermiUpdate calculates correctly":
    require:
      fermiUpdate(Inf, 0.5) == 1.0
      fermiUpdate(-Inf, 0.5) == 0.0
      fermiUpdate(0.5, Inf) == 0.5

  test "generateSnapshots creates correct sequence":
    # currently does not sub sample
    let snapshots = generateSnapshots(100, 10)
    require:
      snapshots.len > 0
      snapshots[0] == 0
      snapshots[^1] <= 100

  test "sampleNeighbor returns valid neighbor":
    var state = State(
      agents:
        @[
          Agent(id: 0, neighbors: {1: 1, 2: 1}.toTable, role: "A"),
          Agent(id: 1, neighbors: {0: 1}.toTable, role: "B"),
          Agent(id: 2, neighbors: {0: 1}.toTable, role: "B"),
        ],
      rng: initRand(42),
    )

    let neighbor = state.sampleNeighbor(0)
    check(neighbor in [0, 1, 2])

  test "changeStrategy flips agent state":
    var agent = Agent(state: 0.0)
    agent.changeStrategy()
    check(agent.state == 1.0)
    agent.changeStrategy()
    check(agent.state == 0.0)

  test "step function":
    var state = State(config: mockConfig)
    let a = {"B": 1.0}.toTable
    let b = {"A": 1.0}.toTable
    state.valueNetwork = {"A": b, "B": a}.toTable
    discard makeAgent(0, state)
    discard makeAgent(1, state)
    state.agents[0].addEdge(state.agents[1])

    var mutations: seq[Mutation] = @[]
    state.step(0, mutations)
    check:
      mutations.len >= 0

  test "simulate function":
    var state = State(config: mockConfig)
    let a = {"B": 1.0}.toTable
    let b = {"A": 1.0}.toTable
    state.valueNetwork = {"A": b, "B": a}.toTable
    var a1 = makeAgent(0, state)
    var a2 = makeAgent(1, state)
    state.agents[0].addEdge(state.agents[1])

    let results = state.simulate(10, 2)
    check:
      results.len == 10
      results[0].len == 2
      a1.parent == a2.parent
      state.agents[0].parent == state.agents[1].parent
      state.agents[0].parent == a1.parent

  test "Check Number of Organizations Simple":
    var state = State(config: mockConfig)
    let a = {"A": 1.0}.toTable
    let b = {"B": 1.0}.toTable
    state.valueNetwork = {"A": b, "B": a}.toTable
    var
      a1 = makeAgent(0, state)
      a2 = makeAgent(1, state)
    a1.state = 1.0
    a2.state = 1.0
    a1.role = "A"
    a2.role = "B"
    a1.addEdge(a2)

    let counts = a1.getCountOrganizations()
    check counts.gangs == 1
    check counts.firms == 1

  test "Check Number of Organizations":
    var state = State(config: mockConfig)
    let a = {"A": 1.0, "C": 1.0}.toTable
    let b = {"B": 1.0, "C": 1.0}.toTable
    let c = {"A": 1.0, "B": 1.0}.toTable
    state.valueNetwork = {"A": b, "B": a, "C": c}.toTable
    var
      a0 = makeAgent(0, state)
      a1 = makeAgent(1, state)
      a2 = makeAgent(2, state)
      a3 = makeAgent(3, state)
    # set all to criminals
    for a in state.agents:
      a.state = 1.0

    # Test this graph --> should be 2 organizations
    #        - 2
    #0 - 1 -
    #        - 3
    a0.role = "A"
    a1.role = "B"
    a2.role = "C"
    a3.role = "C"

    a0.addEdge(a1)
    a1.addEdge(a2)
    a1.addEdge(a3)

    # test all agents
    var solutions = @[(2, 2), (2, 2), (1, 1), (1, 1)]
    for (a, sol) in zip(state.agents, solutions):
      let nums = a.getCountOrganizations()
      echo (a.id, (nums.gangs, nums.firms), sol)
      require (nums.gangs, nums.firms) == sol

    let nums = state.agents[1].getCountOrganizations(maxDepth = 1)
    echo (nums.gangs, nums.firms)

  test "Check number of Organizations":
    var state = State(config: mockConfig)
    let a = {"B": 1.0, "C": 1.0}.toTable
    let b = {"A": 1.0, "C": 1.0}.toTable
    let c = {"A": 1.0, "B": 1.0}.toTable
    state.valueNetwork = {"A": a, "B": b, "C": c}.toTable
    var
      a0 = makeAgent(0, state)
      a1 = makeAgent(1, state)
      a2 = makeAgent(2, state)
      a3 = makeAgent(3, state)
      a4 = makeAgent(4, state)
    # set all to criminals
    for a in state.agents:
      a.state = 1.0

    # Test this graph --> should be 2 organizations
    #        - 2
    #0 - 1 -
    #        - 3 - 4
    a0.role = "A"
    a1.role = "B"
    a2.role = "C"
    a3.role = "C"
    a4.role = "C"

    a0.addEdge(a1)
    a1.addEdge(a2)
    a1.addEdge(a3)
    a3.addEdge(a4)
    let verbose = false

    # if 4 has role C it has no organizations
    # test all agents
    var solutions = @[(2, 2), (2, 2), (1, 1), (1, 1), (0, 0)]
    for (a, sol) in zip(state.agents, solutions):
      let nums = a.getCountOrganizations()
      echo (a.id, (nums.gangs, nums.firms), sol)
      check (nums.gangs, nums.firms) == sol

    a4.role = "B"
    a4.state = 1.0
    for (a, sol) in zip(state.agents, solutions):
      let nums = a.getCountOrganizations()
      echo (a.id, (nums.gangs, nums.firms), sol)
      require (nums.gangs, nums.firms) == sol

    a4.role = "A"
    a4.state = 1.0
    solutions = @[(2, 2), (3, 3), (1, 1), (2, 2), (1, 1)]
    for (a, sol) in zip(state.agents, solutions):
      let nums = a.getCountOrganizations()
      echo (a.id, (nums.gangs, nums.firms), sol)
      require (nums.gangs, nums.firms) == sol
    a4.role = "A"
    a4.state = 1.0
    solutions = @[(2, 2), (3, 3), (1, 1), (2, 2), (1, 1)]
    a4.addEdge(a0)
    for (a, sol) in zip(state.agents, solutions):
      let nums = a.getCountOrganizations()
      if verbose:
        echo (a.id, (nums.gangs, nums.firms), sol)
      require (nums.gangs, nums.firms) == sol

    solutions = @[(0, 2), (0, 0), (0, 1), (0, 2), (0, 1)]
    a1.state = 0.0
    for (a, sol) in zip(state.agents, solutions):
      let nums = a.getCountOrganizations()
      echo (a.id, (nums.gangs, nums.firms), sol)
      require (nums.gangs, nums.firms) == sol

    solutions = @[(2, 2), (2, 3), (1, 1), (1, 2), (0, 0)]
    a1.state = 1.0
    a4.state = 0.0
    for (a, sol) in zip(state.agents, solutions):
      let nums = a.getCountOrganizations()
      echo (a.id, (nums.gangs, nums.firms), sol)
      require (nums.gangs, nums.firms) == sol
      echo state.agents.len

  test "Test VN 4":
    var state = State(config: mockConfig)
    let a = {"B": 1.0, "C": 1.0, "D": 1.0}.toTable()
    let b = {"A": 1.0, "C": 1.0, "D": 1.0}.toTable()
    let c = {"A": 1.0, "B": 1.0, "D": 1.0}.toTable()
    let d = {"A": 1.0, "B": 1.0, "C": 1.0}.toTable()
    var valueNetwork = initTable[string, Table[string, float]]()
    state.valueNetwork = {"A": a, "B": b, "C": c, "D": d}.toTable()
    for idx in (0 .. 4):
      discard makeAgent(idx, state)
    for agent in state.agents:
      agent.state = 1.0
    #        - 2
    #0 - 1 -
    #        - 3 - 4
    state.agents[0].role = "A"
    state.agents[1].role = "B"
    state.agents[2].role = "C"
    state.agents[3].role = "C"
    state.agents[4].role = "D"
    state.agents[0].addEdge(state.agents[1])
    state.agents[1].addEdge(state.agents[2])
    state.agents[1].addEdge(state.agents[3])
    state.agents[2].addEdge(state.agents[4])
    for a in state.agents:
      let nums = a.getCountOrganizations()
      check nums.gangs == 0
      check nums.firms == 0

    test "getPayout calculation":
      # Create a single organization of A-B
      var state = State(config: mockConfig)
      let a = {"A": 1.0}.toTable
      let b = {"B": 1.0}.toTable
      state.valueNetwork = {"A": b, "B": a}.toTable

      # First test the criminal network
      discard makeAgent(0, state)
      discard makeAgent(1, state)
      state.agents[0].state = 1.0
      state.agents[1].state = 1.0
      state.agents[0].role = "A"
      state.agents[1].role = "B"
      state.agents[0].addEdge(state.agents[1])

      var payout = state.getPayOff(0)
      let nSamples = state.agents[0].nSamples.float
      let prior_cost = state.config.cost
      state.config.cost = calculateCost(state, 0, prior_cost)
      #check payout == (state.config.benefit - state.config.cost)

      # non-criminal interacting with other non-criminals
      #get nothing
      state.agents[0].state = 0.0

      state.config.cost = prior_cost
      state.config.cost = calculateCost(state, 0, prior_cost)
      payout = state.getPayoff(0)

      # criminal interacting with non-criminals should ncu#no   cost if their only connection is non-criminal
      state.agents[0].state = 1.0
      state.agents[1].state = 0.0

      state.config.cost = prior_cost
      state.config.cost = calculateCost(state, 0, prior_cost)
      #check state.config.cost == 0.0
      payout = state.getPayoff(0) # should be the same asthecost
      # should be zero as the only connection is on-criminal
      # so the criminal degree is non-criminal
      #check payout == -state.config.cost

      discard makeAgent(2, state)
      state.agents[0].addEdge(state.agents[2])

      state.agents[1].state = 1.0
      state.agents[2].state = 1.0
      state.agents[2].role = "B"
      state.config.cost = prior_cost
      state.config.cost = calculateCost(state, 0, prior_cost)
      payout = state.getPayoff(0)
      # 2 organizations so we get one benefit and 2 egatives

      # the agent should be able to make two (criminal)rganizations
      # check payout == state.config.benefit * 2 - 2 * state.config.cost

  test "Check Dumbell With Shared Center":
    # 4 -       - 2
    #     0 - 1
    # 5 -  \ /  - 3
    #       6
    # In this graph node 0 would be able to make 3 organizations
    # Node 1 would be able to make 2 organizations (excluding 6)
    # All other nodes can make 1 organization (shared in the center)
    var state = State(config: mockConfig)
    let a = {"A": 1.0, "C": 1.0}.toTable
    let b = {"B": 1.0, "C": 1.0}.toTable
    let c = {"A": 1.0, "B": 1.0}.toTable
    state.valueNetwork = {"A": b, "B": a, "C": c}.toTable
    for idx in 0 .. 6:
      discard makeAgent(idx, state)
      state.agents[idx].state = 1.0
    # create the adj
    state.agents[0].addEdge(state.agents[1])
    state.agents[1].addEdge(state.agents[2])
    state.agents[1].addEdge(state.agents[3])
    state.agents[0].addEdge(state.agents[4])
    state.agents[0].addEdge(state.agents[5])
    state.agents[0].addEdge(state.agents[6])
    state.agents[6].addEdge(state.agents[6])

    state.agents[0].role = "A"
    state.agents[1].role = "B"
    state.agents[2].role = "C"
    state.agents[3].role = "C"
    state.agents[4].role = "C"
    state.agents[5].role = "C"
    state.agents[6].role = "C"
    var solutions = [(5, 5), (5, 5), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    for (a, sol) in zip(state.agents, solutions):
      let nums = a.getCountOrganizations()
      check nums.gangs == sol[0]
      check nums.firms == sol[1]
