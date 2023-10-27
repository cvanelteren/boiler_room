import sequtils, random, math, strutils,
            strformat, strutils, tables, sequtils, terminal
import utils
import nimpy


type Agent* = ref object
  id*: int
  state*: float
  role*: string
  neighbors*: Table[string, seq[ptr Agent]]
  bias*: float
  n_samples*: int
  trust*: ptr seq[seq[float]]

type State* = object of Config
  agents*: seq[Agent]
  rng*: Rand
  p*: Table[string, Table[float, float]]
  trust*: seq[seq[float]]

proc get*(agents: seq[ref Agent]): seq[int] =
  result = agents.mapIt(it.id)

proc makeAgent*(id: int, state: State): Agent =
  result = Agent(id: id,
                 neighbors: initTable[string, seq[ptr Agent]](),
                 role: state.roles.sample,
                 state: state.states.sample,
                 bias: 0.0,
                 n_samples:  state.n_samples,
                 )

proc add_neighbor*(this: var Agent, other: var Agent, directed = false) =
  if this.neighbors.hasKeyOrPut(other.role, @[other.addr]):
    this.neighbors[other.role].add other.addr
  if not directed:
    if other.neighbors.hasKeyOrPut(this.role, @[this.addr]):
      other.neighbors[this.role].add this.addr

proc makeNetwork*(state: var State, g: PyObject) =
  # add agents
  for node in g.nodes():
    let node_defaults = g.nodes[node].to Table[string, PyObject]
    let agent = makeAgent(id = node.to(int), state = state)

    # assign a role
    if "role" in node_defaults:
      agent.role = node_defaults["role"].to string
    else:
      agent.role =  state.rng.sample(state.roles, state.p_roles.cumsummed)

    # assign a state
    if "state" in node_defaults:
      agent.state = node_defaults["state"].to float
    else:
      var p: seq[float]
      for (role, pi) in state.roles.zip(state.p_states):
        if role == agent.role:
          p = pi
          break
      agent.state = state.rng.sample(state.states, p.cumsummed)

    # assign a bias
    agent.bias = 0.0
    if "bias" in node_defaults:
      agent.bias = node_defaults["bias"].to float

    # assign an explorg rate
    state.agents.add agent

  for edge in g.edges():
    let x = edge[0].to int
    let y = edge[1].to int
    if x == y:
      continue
    state.agents[x].add_neighbor(state.agents[y],
                                 directed = g.is_directed().to bool)

proc makeState*(config: Config): State =
  # TODO: This way is error prone as the
  # default would be zeros for a value
  result = State(states: config.states,
                 roles: config.roles,
                 benefit: config.benefit,
                 cost: config.cost,
                 beta: config.beta,
                 alpha: config.alpha,
                 rng: initRand(config.seed),
                 p_states: config.p_states,
                 p_roles: config.p_roles,
                 z: config.z,
                 t: config.t,
                 seed: config.seed,
                 n_samples: config.n_samples,
                 agents: @[],
                 trial: config.trial,
                 rewire: config.rewire,
                 depth: config.depth,
                 mu: config.mu,
  )
  # add agents and connections
  result.makeNetwork(config.g)
  result.trust = newSeqWith(result.agents.len,
                            newSeqWith( result.agents.len, 1.0)
  )

  for agent in result.agents:
    agent.trust = result.trust.addr


proc energy(agent: Agent, interactions: seq[float], state: State): float =
  result = state.benefit * interactions.prod -
    state.cost * interactions[0]


proc fermiUpdate*(delta, beta: float): float {.exportpy.} =
  result = 1.0 / (1.0 + exp(-beta * delta))

proc getPayoffDifference(agent: Agent,
                         interactions: var seq[float],
                         state: var State): float =
  # compute energy difference (payoff difference)
  var currentEnergy = agent.energy(interactions, state)
  proc filter(this, other: float, n: int): float =
    result = 0.0
    if this != other:
      result = 1/(n - 1)

  let ps = state.states.mapIt(
    filter(it, agent.state, state.states.len)
  )
  interactions[0] = state.rng.sample(state.states, ps.cumsummed)
  assert interactions[0] != agent.state
  var flipEnergy = agent.energy(interactions, state)
  result = flipEnergy - currentEnergy

# how many time to sample... sample until you get a hit?
proc sample(agent: Agent,
            state: var State,
            order: int,
            sampled: var seq[ptr Agent]): float =
  # fermi update
  var
    interactions: seq[float] = newSeqWith(order, 0.0)
    idx = 1 # skip first index; only interested in neighborstates
  interactions[0] = agent.state
  # Create distribution of surrounding agent roles and states
  # let cdf = agent.sense()
  # echo cdf
  for role, neighbors in agent.neighbors:
    if role != agent.role:
      # let weights = neighbors.mapIt(agent.trust[agent.id][it.id] + 1e-6).cumsummed()
      # var other: ptr Agent = state.rng.sample(neighbors, weights)
      let other = state.rng.sample(neighbors)
      interactions[idx] = other.state
      sampled.add other
      idx.inc
  result = agent.getPayoffDifference(interactions, state)



proc update(state: var State, id: int, order = 3) =
  if order < 1:
    raise (ref ValueError)(msg: "Order cannot be smaller than 1")
  let agent = state.agents[id]
  var delta = 0.0
  let z = 1/(agent.n_samples.float)
  var sampled: seq[ptr Agent] = @[]
  for sample in 0..<agent.n_samples:
    delta += agent.sample(state, order, sampled)
  let p  = fermiUpdate(delta * z, state.beta)

  if state.rng.rand(1.0) < p:
    if agent.state == 1.0:
      agent.state = 0.0
    else:
      agent.state = 1.0

  # # update trust levels
  # let zz = (agent.n_samples.float * (state.roles.len - 1).float)
  # for other in sampled:
  #   var gain = 1.0/zz
  #   if agent.state != other.state:
  #     gain *= -1.0
  #   agent.trust[agent.id][other.id] += gain
  #   agent.trust[other.id][agent.id] += gain
  #   if agent.trust[agent.id][other.id] <= 0.0:
  #     agent.trust[agent.id][other.id] = 0.0
  #   if agent.trust[other.id][agent.id] <= 0.0:
  #     agent.trust[other.id][agent.id] = 0.0


proc simulate*(state: var State, t: int): seq[State] =
  # keep diffs for copy
  var agents = (0..<state.agents.len).toseq
  result = newSeq[State](t)
  var count = initCountTable[float]()
  for agent in state.agents:
    count.inc(agent.state)
  for ti in 0..<t:
    result[ti] = state.deepcopy()

    for tmp in agents:
      let agent = state.rng.sample(agents)
      state.update(agent)

proc `echo`*(config: Config) =
  echo '-'.repeat(16), " Parameters ", '-'.repeat(16)
  for key, value in config.fieldPairs():
    echo key, ": ", value
