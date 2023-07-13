import std/[unittest, math, strutils]
import boiler_room/agent
import nimpy

suite "Test python parse":
    setup:
      # required
      let nx = pyImport("networkx")

    test "Make simulation":
      let g = nx.complete_graph(10)
      let config = Config(
                g: g,
                roles: "A B C".split(),
                states: @[1.0],
                benefit: 1.0,
                cost: 1.0,
                beta: 1.0)

suite "Fermi update":
    test "Dealing with infinities":
        var delta: float
        delta = Inf
        check fermiUpdate(delta, 1.0) == 1.0

        delta = NegInf
        check fermiUpdate(delta, 1.0) == 0.0

    test "Equal energies":
        check fermiUpdate(0.0, 1.0) == 0.5
