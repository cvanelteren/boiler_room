import unittest, math, strutils, random, tables, sequtils
import nimpy, sugar
import boiler_room
from boiler_room import agent
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
      agent = Agent(id: 0, neighbors: initTable[string, seq[ptr Agent]]())
      let n = 10
      var agents = newSeq[Agent](n)
      for id in 1..n:
        var other = Agent(id: id, role: "test")
        agents[id - 1] = other
        agent.add_neighbor(agents[id-1])

      agent.neighbors["test"].shuffle
      for role, neighbors in agent.neighbors:
        for neighbor in neighbors:
          check neighbor.isnil == false
          check neighbor.role == "test"

      check agent.neighbors["test"].sample().isnil == false
      check agent.neighbors["test"].sample().id in agent.neighbors["test"].mapIt(it.id)
