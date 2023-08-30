import unittest, math, strutils, random, tables, sequtils
import nimpy, sugar
import boiler_room
from boiler_room import agent
import parsetoml

suite "Test python parse":
    setup:
      # required
      let nx = pyImport("networkx")

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
      var agents: seq[Agent] = @[]
      var ids: seq[int] = @[]
      for tmp in 0..10:
        let other = Agent(id: agent.id + tmp + 2)

        ids.add other.id
        agents.add(other)
      for idx, other in agents:
        agent.neighbors.add(agents[idx].addr)
      agent.neighbors.shuffle()
      check ids != agent.neighbors.mapIt(it.id)


  # test "Checking incomplete inputs":
    # config = "./tests/default.toml".read(target = "test override")
    # expect newException(ValueError, "Ratio cannot be computed, needs both @c and @b")
