import sequtils, random, math, strutils,
            strformat, strutils, tables, sequtils, terminal, options
import sets
import utils
import nimpy
let np = pyImport("numpy")

type Agent* = ref object
  id*: int
  state*: float
  role*: string
  neighbors*: Table[int, int]
  bias*: float
  n_samples*: int
  edgeRate*: float
  mutationRate*: float

type Mutation* = object
    id*: int
    state*: float
    neighbors*: Table[int, int]
    role*: string

proc hasNeighbor(this: Agent, other: int): bool =
  result = other in this.neighbors

type State* = object of Config
  agents*: seq[Agent]
  rng*: Rand
  p*: Table[string, Table[float, float]]


proc makeAgent*(id: int, state: State): Agent =
  result = Agent(id: id,
                 neighbors: initTable[int, int](),
                 role: state.roles.sample,
                 state: state.states.sample,
                 bias: 0.0,
                 n_samples:  state.n_samples,
                 edgeRate: 0.0,
                 mutationRate: 0.0
                 )

proc addEdge*(this: var Agent, other: var Agent, directed = false) =
  if this == other:
    return
  this.neighbors[other.id] = 1
  other.neighbors[this.id] = 1


proc rmEdge*(this, other: var Agent, directed = false) =
  # remove edge
  if other.id in this.neighbors:
    this.neighbors.del(other.id)
  if this.id in other.neighbors:
    other.neighbors.del this.id


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

    if "edgeRate" in node_defaults:
      agent.edgeRate = node_defaults["edgeRate"].to float


    if "mutationRate" in node_defaults:
      agent.mutationRate = node_defaults["mutationRate"].to float

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
    state.agents[x].addEdge(state.agents[y],
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
                 assortativity: config.assortativity,
                 mu: config.mu,
                 step: config.step,
  )
  # add agents and connections
  result.makeNetwork(config.g)

proc energy(agent: Agent, interactions: seq[float], state: State): float =
  result = state.benefit * interactions.prod - state.cost * interactions[0]


proc fermiUpdate*(delta, beta: float): float =
  result = 1.0 / (1.0 + exp(-(1/beta) * delta))

proc sample(agent: Agent,
            state: var State,
            order: int,
            ): float =
  # fermi update
  var interactions = newSeqWith(order, 0.0)
  interactions[0] = agent.state
  # Create distribution of surrounding agent roles and states
  var seen = @[agent.role]
  var attempt = 1
  if agent.neighbors.len == 0:
    return 0.0
  var neighbors = agent.neighbors.keys.toseq()
  # neighbors.shuffle()
  var other: int
  for attempt in 1..<order:
    other = state.rng.sample(neighbors)
    if state.agents[other].role notin seen:
      seen.add state.agents[other].role
      interactions[attempt] = state.agents[other].state
      # attempt.inc
  result = agent.energy(interactions, state)


proc getPayout(state: var State, id: int, order = 3): float =
  if order < 1:
    raise (ref ValueError)(msg: "Order cannot be smaller than 1")
  var agent = state.agents[id]
  for sample in 0..<agent.n_samples:
    result += agent.sample(state, order)

proc makeMutation(agent: Agent): Mutation =
  result = Mutation(
      id: agent.id,
      neighbors: agent.neighbors,
      state: agent.state,
      role: agent.role
    )

proc generateSnapshots(t, n: int): seq[int] =
  return (0..<t).toseq()
  if n <= 0:
    return (0..<t).toseq()
  let first = (0.5 * n.float).int
  let second = (0.50 * n.float).int
  let m = (t - first).div(second)
  # echo &"{first=} {second=} {n=} {m=}"
  result = (1..<first).toseq().concat(countUp(first, t, m).toseq())


proc sampleNeighbor(state: var State, agent: int): int =
  let agent = state.agents[agent]
  if agent.neighbors.len == 0 or state.rng.rand(1.0) < agent.mutationRate:
    var other = state.rng.sample(state.agents).id
    while other == agent.id and state.agents[other].role != agent.role:
      other = state.rng.sample(state.agents).id
    return other

  # sample a random neighbor of neighbors
  result = state.rng.sample(agent.neighbors.keys().toseq())
  result = state.rng.sample(state.agents[result].neighbors.keys().toseq())

proc step(state: var State, agent: int, mutations: var seq[Mutation]) =
  # determine which to perform
  var currents = @[state.agents[agent].makeMutation()]
  var buffer = [0.0, 0.0]

  if state.rng.rand(1.0) < state.agents[agent].edgeRate:
    let other = state.sampleNeighbor(agent)


    # let bprior = state.benefit
    let prior = state.cost
    state.cost = prior * state.agents[agent].neighbors.len.float
    # state.benefit = bprior * state.agents[agent].neighbors.len.float
    buffer[0] = state.getPayout(agent, order = state.roles.len)
    # buffer[0] += state.getPayout(other, order = state.roles.len)

    currents.add state.agents[other].makeMutation()
    # consider opposite of current state
    if state.agents[agent].hasNeighbor(other):
      state.agents[agent].rmEdge(state.agents[other])
    else:
      state.agents[agent].addEdge(state.agents[other])

    state.cost = prior * state.agents[agent].neighbors.len.float
    # state.benefit = bprior * state.agents[agent].neighbors.len.float
    # let zz = (1/state.agents[agent].n_samples.float) *
    buffer[1] = state.getPayout(agent, order = state.roles.len)
    # buffer[1] += state.getPayout(other, order = state.roles.len)
    state.cost = prior
    # state.benefit = bprior

  else:
    # TODO: make more general
    let z = 1/(state.agents[agent].n_samples.float)
    buffer[0] = z * state.getPayout(agent, order = state.roles.len)
    var newState = 1.0
    if state.agents[agent].state == newState:
      newState = 0.0
    state.agents[agent].state = newState
    buffer[1] = z * state.getPayout(agent, order = state.roles.len)

  let delta = (buffer[1] - buffer[0])
  # echo (state.cost, state.beta, delta, fermiUpdate(delta, state.beta))
  # accept with fermi-rule
  # echo fermiUpdate(delta, state.beta), (state.beta, state.cost)
  if state.rng.rand(1.0) < fermiUpdate(delta, state.beta):
    for idx, current in currents:
      mutations.add state.agents[current.id].makeMutation()
        # let msg = &"{tmp[idx]}, {state.agents[current.id].neighbors.len} {delete}"
        # assert state.agents[current.id].neighbors.len != (tmp[idx] - 1), msg
  # reject new state
  else:
    for current in currents:
      state.agents[current.id].state = current.state
      state.agents[current.id].neighbors = current.neighbors
      state.agents[current.id].role = current.role


proc simulate*(state: var State, t: int, n: int = 0): seq[seq[Mutation]] =
  # keep diffs for copy
  var agents = (0..<state.agents.len).toseq

  var snapshots = generateSnapshots(t, n)
  result = newSeq[newSeq[Mutation]()](snapshots.len)
  # result = newSeq[newSeq[Mutation]()](t)
  # NOTE: mutations are now stored dense in its adjacency structure
  # since we are having additions and removals, we need to keep track
  # of both which could be remapped to positive or negative indices
  # to save on memory.

  var mutations = state.agents.mapIt(it.makeMutation)
  var snap = 0
  for ti in 0..<t:
    if ti == snapshots[snap]:
      result[snap] = mutations
      snap.inc


    mutations = @[]
    agents.shuffle()
    # let agent = state.rng.sample(agents)
    # state.step(state.rng.sample(agents), mutations)
    for agent in agents:
      #NOTE: will add to mutations if new state is accepted
      step(state, agent, mutations)

  # var s = 0.0
  # for agent in state.agents:
  #   s += agent.state / state.agents.len.float
  # echo " "
  # echo (s, state.beta, state.cost)



proc `echo`*(config: Config) =
  echo '-'.repeat(16), " Parameters ", '-'.repeat(16)
  for key, value in config.fieldPairs():
    echo key, ": ", value
