import std/[unittest, math, strutils, random, tables, sequtils]
import nimpy
import boiler_room.agent

suite "Test python parse":
    setup:
      # required
      let nx = pyImport("networkx")

    test "Make simulation":
      let g = nx.complete_graph(10)
      let config = Config(
                roles: "A B C".split(),
                states: @[1.0],
                benefit: 1.0,
                cost: 1.0,
                beta: 1.0)
      # makeSimulation(config)

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

  test "Trust sampling":
    # make some agents with trust updating
    # agents with opposite state will have their trust decreased
    let neighbors = @[Agent(id: 1, state: 0.0), Agent(id: 2, state: 1.0)]
    let trust: Table[int, float] = neighbors.mapIt((it.id, 1.0)).toTable
    agent = Agent(state: 0.0, id: 1,
                  neighbors: neighbors.mapIt(it.addr),
                  trust: trust)
    var rng = initRand()
    check agent.getTrustCDF() == @[1.0, 2.0]


    # check update trust
    agent.updateTrust(0.5)
    check agent.trust == {1: 1.5, 2: 0.5}.toTable

    # check sampling ratio
    var ids = initTable[int, int]()
    for idx in 0..<100000:
      let s = rng.sample(agent.neighbors, cdf = agent.getTrustCDF()).id
      if ids.haskeyorput(s, 1):
        ids[s].inc
    let threshold = 0.05
    check (ids[1] / ids[2] - 3.0 <= threshold)
