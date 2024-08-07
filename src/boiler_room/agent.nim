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


proc toAdj*(state: State): seq[(int, int)] =
  var unique = initHashSet[(int, int)]()
  for agent in state.agents:
    for neighbor in agent.neighbors.keys():
      unique.incl (agent.id, neighbor)
  result = unique.toseq()


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

    # var idx = 0
    # while idx < 100:
    #   other = state.rng.sample(neighbors)
    #   if state.agents[other].role notin seen:
    #     seen.add state.agents[other].role
    #     interactions[attempt] = state.agents[other].state
    #     break
    #   idx.inc

    other = state.rng.sample(neighbors)
    if state.agents[other].role notin seen:
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
  if agent.neighbors.len > 0:
    result = state.rng.sample(agent.neighbors.keys().toseq())
    if state.agents[result].neighbors.len > 0:
      return state.rng.sample(state.agents[result].neighbors.keys().toseq())
  # default option
  return agent.id

proc step(state: var State, agent: int, mutations: var seq[Mutation]) =
  # determine which to perform
  var currents = @[state.agents[agent].makeMutation()]
  var buffer = [0.0, 0.0]

  let Z = 1.0/(state.agents.len - 1).float
  let prior = state.cost
  let pp = state.benefit
  if state.rng.rand(1.0) < state.agents[agent].edgeRate:
    let other = state.sampleNeighbor(agent)


    # let bprior = state.benefit
    var z = 1/(state.agents[agent].neighbors.len.float)

    # var tmp_cost = state.agents[agent].neighbors.len.float

    var tmp_cost = 0.0
    for neighbor in state.agents[agent].neighbors.keys():
      if state.agents[neighbor].state == 1.0:
        tmp_cost += 1.0

    state.cost = Z * tmp_cost * prior
    buffer[0] = state.getPayout(agent, order = state.roles.len)

    currents.add state.agents[other].makeMutation()
    # consider opposite of current state
    if state.agents[agent].hasNeighbor(other):
      state.agents[agent].rmEdge(state.agents[other])
    else:
      state.agents[agent].addEdge(state.agents[other])

    # tmp_cost = state.agents[agent].neighbors.len.float

    tmp_cost = 0.0
    for neighbor in state.agents[agent].neighbors.keys():
      if state.agents[neighbor].state == 1.0:
        tmp_cost += 1.0

    # z = 1/(state.agents[agent].neighbors.len.float)
    state.cost = Z * tmp_cost * prior

    buffer[1] = state.getPayout(agent, order = state.roles.len)

  else:
    # TODO: make more general

    # get number of criminal neighbors
    # var tmp_cost = state.agents[agent].neighbors.len.float

    var tmp_cost = 0.0
    for neighbor in state.agents[agent].neighbors.keys():
      if state.agents[neighbor].state == 1.0:
        tmp_cost += 1.0
    # let z = 1/(state.agents[agent].neighbors.len.float)

    state.cost = Z * tmp_cost * prior
    buffer[0] = state.getPayout(agent, order = state.roles.len)
    # change strategy
    var newState = 1.0
    if state.agents[agent].state == newState:
      newState = 0.0
    state.agents[agent].state = newState
    buffer[1] = state.getPayout(agent, order = state.roles.len)

  let delta = (buffer[1] - buffer[0])
  # echo &"{state.cost=} {fermiUpdate(delta,state.beta)=}"
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
  # echo &"{state.cost=} {prior=}"
  state.cost = prior
  state.benefit = pp


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

proc makeBuffer(n: int,
                state: var State,
                agents: var seq[int],
                mutations: var seq[Mutation]): seq[float] =

  result = newSeqWith[float](n, 1.0)
  let z = 1/n.float
  # fill the buffer
  for idx in 0..<result.len:
    result[idx] = state.agents.foldl(a + b.neighbors.len,
                                     state.agents[0].neighbors.len).float * z
    # perform a step
    agents.shuffle()
    mutations = @[] # empty mutations to prevent blow up
    for agent in agents:
      step(state, agent, mutations)

proc findEquilibrium(state: var State,
                     agents: var seq[int],
                     mutations: var seq[Mutation],
                     threshold: float) =

  var buffer = makeBuffer(100, state, agents, mutations)
  let nbuf = buffer.len
  let z = 1/nbuf.float
  var zz = state.agents.len.float
  zz = 1/ (zz * (zz - 1) / 2.0)
  var idx = 0
  # equilibrate
  proc mse(buffer: seq[float]): float =
    for idx, other in buffer[1..^1]:
      result += (other - buffer[idx])^2
    result *= 1/(buffer.len.float)

  idx = 0
  while mse(buffer) > threshold:
    buffer[idx] = state.agents.foldl(a + b.neighbors.len,
                  state.agents[0].neighbors.len).float * zz
    idx = (idx + 1).mod(nbuf)
    # perform a step
    agents.shuffle()
    mutations = @[] # empty mutations to prevent blow up
    for agent in agents:
      step(state, agent, mutations)


proc simulateInEquilibrium*(state: var State, n = 0, threshold = 1e-2, mutationAfter = 1.0): seq[seq[Mutation]] =
  assert n >= 1
  var agents = (0..<state.agents.len).toseq()
  # NOTE: the first index is the startin state and should contain all the agents
  result = newSeq[newSeq[Mutation]()](n)
  var mutations = state.agents.mapIt(it.makeMutation)
  result[0] = mutations

  # we equilibrate in the edge change
  var z = state.agents.len.float
  z  = z * (z - 1) / 2


  state.findEquilibrium(agents, mutations, threshold)
  for agent in state.agents:
    agent.mutationRate = mutationAfter

  # sample from the equilibrium
  for sample in 1..<n:
    result[sample] = mutations
    # perform a step
    agents.shuffle()
    mutations = @[] # empty mutations to prevent blow up
    for agent in agents:
      step(state, agent, mutations)


proc `echo`*(config: Config) =
  echo '-'.repeat(16), " Parameters ", '-'.repeat(16)
  for key, value in config.fieldPairs():
    echo key, ": ", value
