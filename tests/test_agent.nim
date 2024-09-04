import unittest
import sequtils, random, tables, math
import nimpy
import boiler_room
#import boiler_room.agent # Assuming the code you provided is in a file named agent.nim

# Mock PyObject for testing
type MockPyObject = ref object

suite "Agent Module Tests":
  setup:
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

  test "makeAgent creates an Agent with correct properties":
    var state = State(config: mockConfig, rng: initRand(42))
    let agent = makeAgent(0, state)
    let other = makeAgent(1, state)

    check:
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

    check:
      agent1.neighbors.hasKey(2)
      agent2.neighbors.hasKey(1)

  test "rmEdge removes edge correctly":
    var agent1 = Agent(id: 1, neighbors: {2: 1}.toTable)
    var agent2 = Agent(id: 2, neighbors: {1: 1}.toTable)

    agent1.rmEdge(agent2)

    check:
      not agent1.neighbors.hasKey(2)
      not agent2.neighbors.hasKey(1)

  test "fermiUpdate calculates correctly":
    check:
      fermiUpdate(Inf, 0.5) == 1.0
      fermiUpdate(-Inf, 0.5) == 0.0
      fermiUpdate(0.5, Inf) == 0.5

  test "generateSnapshots creates correct sequence":
    # currently does not sub sample
    let snapshots = generateSnapshots(100, 10)
    check:
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

  test "getPayout calculation":
    var state = State(config: mockConfig)
    let a = {"A": 1.0}.toTable
    let b = {"B": 1.0}.toTable
    state.valueNetwork = {"A": b, "B": a}.toTable

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
    check:
      payout == nSamples * (state.config.benefit - state.config.cost)

    # non-criminal interacting with other non-criminals
    #get nothing
    state.agents[0].state = 0.0

    state.config.cost = prior_cost
    state.config.cost = calculateCost(state, 0, prior_cost)
    payout = state.getPayOff(0)
    check payout == 0.0
    # criminal interacting with non-criminals should incur no cost if their only connection is non-criminal
    state.agents[0].state = 1.0
    state.agents[1].state = 0.0

    state.config.cost = prior_cost
    state.config.cost = calculateCost(state, 0, prior_cost)
    check state.config.cost == 0.0
    payout = state.getPayOff(0)
    # should be zero as the only connection is non-criminal
    # so the criminal degree is non-criminal
    check payout == nSamples * state.config.cost

    discard makeAgent(2, state)
    state.agents[0].addEdge(state.agents[2])
    state.agents[2].state = 1.0
    state.agents[2].role = "B"
    state.config.cost = calculateCost(state, 0, prior_cost)
    check state.config.cost > 0.0
    payout = state.getPayOff(0)

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

  test "check number of organizations":
    var state = State(config: mockConfig)
    let a = {"B": 1.0}.toTable
    let b = {"A": 1.0}.toTable
    state.valueNetwork = {"A": b, "B": a}.toTable
    var
      a1 = makeAgent(0, state)
      a2 = makeAgent(1, state)
    # set all to criminals
    a1.state = 1.0
    a2.state = 1.0
    # test for 1
    a1.addEdge(a2)

    let availableRoles = a1.getAvailableRoles()
    let nums = state.getPayoff(a1.id)
    # should be succesfull every time
    check nums == -5.0
