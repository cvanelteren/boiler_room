import unittest, tables, sequtils, random
from boiler_room import agent, utils
from boiler_room.utils import read
from boiler_room.agent import makeState, simulate



import nimpy
let nx = pyImport("networkx")
let np = pyImport("numpy")
proc makeBlock*(sizes: seq[int], p: float): PyObject =
  var pi = np.eye(sizes.len)
  pi[0][1] = p
  pi[1][0] = p
  pi[1][2] = p
  pi[2][1] = p
  pi[2][0] = p
  pi[0][2] = p
  result = nx.stochastic_block_model(sizes = sizes, p = pi)


suite "Test state":
  # make a dummy setup from toml file and check the
  # stats of the agents
  setup:
    var config = "tests/test_state.toml".read()
    let sizes = (1..config.roles.len).toseq.mapIt(config.z.div config.roles.len)
    var g = makeBlock(sizes, 1.0)
    for node in g.nodes():
      let group = g.nodes[node]["block"].to int
      g.nodes[node]["role"] = config.roles[group]
    config.g = g
    var state = makeState(config)

    # utils
    var state_map = initTable[float, int]()
    for idx, state in config.states:
        state_map[state] = idx

    var role_map = initTable[string, int]()
    for idx, role in config.roles:
      role_map[role] = idx

    var z: seq[float]
    for p in config.p_roles:
      z.add p * config.z.float

  test "Confirming stats":
      var roles = initTable[string, float]()
      var p_state = newSeqWith(config.roles.len,
            newSeqWith(config.states.len, 0.0))

      for agent in state.agents:
        if roles.haskeyorput(agent.role, 1.0):
            roles[agent.role] += 1.0

        let idx = role_map[agent.role]
        let jdx = state_map[agent.state]
        p_state[idx][jdx] += 1.0 / z[idx]

      # checking role fractions
      var idx = 0
      for key, value in roles:
        check (value.float/state.z.float - config.p_roles[idx]) < 0.01
        idx.inc

      # checking state fractions
      for (p, true_p) in p_state.zip(config.p_states):
        for (pi, true_pi) in p.zip(true_p):
          check (pi - true_pi).abs < 0.05

  test "Sampling distribution agents":
    let n = 1000000
    var agents = initTable[int, float]()
    for idx in 0..<n:
      let id = state.rng.sample(state.agents).id
      if agents.hasKeyOrPut(id, 1.0/n.float):
        agents[id] += 1.0/n.float

    check agents.len == state.agents.len
    for k, v in agents:
      check abs(v - 1/state.agents.len.float) < 0.001

  test "Test agent reference":
    let sample = state.agents[0].neighbors["Production"][^1]
    state.agents[sample.id].state = 3.0
    check state.agents[sample.id] == sample[]
    check state.agents[sample.id].state == sample.state

  # test "Adding edges":
  #     var prop = state.agents[0]
  #     var other = prop.neighbors["Distribution"][0]
  #     var numberOfNeighbors = prop.neighbors["Distribution"].len
  #     prop.rmEdge(state.agents[prop.id])
  #     assert prop.neighbors["Distribution"].len == numberOfNeighbors - 1
  #     prop.addEdge(state.agents[prop.id])
  #     assert prop.neighbors["Distribution"].len == numberOfNeighbors
