import std/[strformat, strutils, random, sequtils], nimpy
from boiler_room import agent, Config, makeState, simulate

# We use networkx to generate  graphs on which the model can
# run The example below  will create a well-mixed condition;
# well-mixed  population is  nothing  more  than a  complete
# graph where all agents can interact with all other agent
let
  nx = pyImport("networkx")

if isMainModule:
  let n_agents = 100 # number of agents
  let t  = 100 # time to simulate for

  var g = nx.complete_graph(n_agents)

  # to setup the system, we need to tell it what roles there
  # are, and  how to assign them.  There are two ways  to do
  # this, either we provide a probability of an agent having
  # a  role, or  we give  them  the roles  inside the  graph
  # structure, I will opt for the later here.

  let roles = "production distribution management".split()
  let  seed = 1234
  var rng = initRand(seed)
  for node in g.nodes():
    g.nodes[node]["role"] = rng.sample(roles)
  # next we give an initial distribution of the roles,
  # we denote 0.0 for non-criminal, and 1.0 for criminal
  # @p_states, controls the probability of non-criminal (idx = 0),
  # and criminal (idx = 1) for each role ^
  let p_states = @[@[0.5, 0.5], @[0.5, 0.5], @[0.5, 0.5]]

  # the core parameters that can influence the model are
  # beta: the noise parameter
  # benefit: benefit for payout
  # cost: cost for criminal activity
  # See source for what can be controlled
  let config = Config(g: g,
                      t: t,
                      z: g.number_of_nodes().to(int),
                      roles: roles,
                      benefit: 10.0,
                      states: @[0.0, 1.0],
                      p_states: p_states,
                      seed: seed,
                      cost: 1.0)

  var system = makeState(config)
  let output = system.simulate(config.t)
  for t, state in output:
    let s = state.agents.mapIt(it.state)
    echo &"{t=} agents have states {s}"
