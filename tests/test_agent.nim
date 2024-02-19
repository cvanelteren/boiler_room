import unittest, math, strutils, random, tables, sequtils
import nimpy, sugar
import boiler_room
from boiler_room import agent
import boiler_room.agent
import parsetoml

# suite "Test python parse":
#     setup:
#       # required
#       let nx = pyImport("networkx")

    # test "Make simulation":
    #   let g = nx.complete_graph(10)
    #   let config = Config(
    #             roles: "A B C".split(),
    #             states: @[1.0],
    #             benefit: 1.0,
    #             cost: 1.0,
    #             beta: 1.0)
    #   var state = makeState(config, g)
    #   proc test(s: State): State =
    #     result = s.deepcopy
    #   state = test(state)
    #   check not state.isnil
    #   makeNetwork(state, g)


suite "Fermi update":
    test "Dealing with infinities":
        var delta: float
        delta = Inf
        check fermiUpdate(delta, 1.0) == 1.0

        delta = NegInf
        check fermiUpdate(delta, 1.0) == 0.0

    test "Equal energies":
        check fermiUpdate(0.0, 1.0) == 0.5

suite "Agent":
  setup:
    var agent: Agent

  test "Make null agent":
    agent = Agent()
    check not agent.isnil
    check agent.id == 0
    check agent.neighbors.len == 0
    check agent.state == 0

  test "Make agent":
    agent = Agent(state: 0.0, id: 1)
    check agent.id == 1
    check agent.state == 0.0

  test "Creating neighbors":
      agent = Agent(id: 0, neighbors: @[])
      let n = 10
      var agents = newSeq[Agent](n)
      for id in 1..n:
        var other = Agent(id: id, role: "test")
        agents[id - 1] = other
        agent.addEdge(agents[id-1])

      check agent.neighbors.len == n
  test "Deleting neighbors":
      var agents = newSeq[Agent](2)
      for id in 0..1:
        agents[id] = Agent(id: id, neighbors: @[])
      agents[0].addEdge(agents[1])
      check agents[0].neighbors.len == agents[1].neighbors.len
      check agents[0].neighbors.len == 1

      agents[0].rmEdge(agents[1])
      check agents[0].neighbors.len == 0
      check agents[1].neighbors.len == 0

  test "Agent property":
    agent = Agent(n_samples: 3, state: -5.0)
    check agent.n_samples == 3
    check agent.state == -5.0
