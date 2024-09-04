import
  std/[
    sequtils, random, math, strutils, tables, sets, strformat, strutils, tables,
    sequtils, terminal, options,
  ]
import os
import nimpy
let np = pyImport("numpy")
let nx = pyImport "networkx"
let pd = pyImport "pandas"
let pycopy = pyImport "copy"

type
  # Holds the confiruation of the simulation
  Config* = ref object
    beta*, benefit*, cost*: float

    depth*: int
    n_samples*, t*, seed*, z*: int
    trial*, n_trials*: int
    step*: int

    p_states*: Table[float, float]
    p_roles*: Table[string, float]

  Agent* = ref object
    id*: int
    state*: float
    role*: string
    neighbors*: Table[int, int]
    bias*: float
    nSamples*: int
    edgeRate*: float
    mutationRate*: float
    parent*: State

  # Store the mutations of the simulation over time
  # This makes it more memory efficient (potentially)
  Mutation* = object
    id*: int
    state*: float
    neighbors*: Table[int, int]
    role*: string

  # Hold the simulation
  State* = ref object
    agents*: seq[Agent]
    rng*: Rand
    valueNetwork*: Table[string, Table[string, float]]
    config*: Config

  # Convert to a pandas row entry
  DataPoint* =
    tuple[
      states: seq[seq[float]],
      roles: seq[seq[string]],
      benefit, cost, beta: float,
      adj: seq[Table[int, seq[int]]],
      trial: int,
    ]

  SimInfo = ref object
    state*: State
    info*: string
    intervention*: string
    n_intervention*: int

import utils

proc random_role(s: var State): string =
  result = s.rng.sample(
    s.config.p_roles.keys().toseq(), s.config.p_roles.values.toseq.cumsummed()
  )

proc random_state(s: var State): float =
  result = s.rng.sample(
    s.config.p_states.keys().toseq(), s.config.p_states.values().toseq().cumsummed()
  )

proc hasNeighbor*(this: Agent, other: int): bool =
  result = other in this.neighbors

proc toAdj*(state: State): seq[(int, int)] =
  var unique = initHashSet[(int, int)]()
  for agent in state.agents:
    for neighbor in agent.neighbors.keys():
      unique.incl (agent.id, neighbor)
  result = unique.toseq()

proc makeAgent*(id: int, state: var State): Agent =
  for agent in state.agents:
    if agent.id == id:
      raise newException(ValueError, &"Agent with id {id} already exists")
  result = Agent(
    id: id,
    neighbors: initTable[int, int](),
    role: state.random_role(),
    state: state.random_state(),
    bias: 0.0,
    nSamples: state.config.n_samples,
    edgeRate: 0.0,
    mutationRate: 0.0,
    parent: state,
  )
  state.agents.add result

proc addEdge*(this: var Agent, other: var Agent, directed = false) =
  if this.id == other.id:
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
    var agent = makeAgent(id = node.to(int), state = state)
    assert state == agent.parent
    # assign a role
    if "role" in node_defaults:
      agent.role = node_defaults["role"].to string
    else:
      agent.role = state.random_role()

    # assign a state
    if "state" in node_defaults:
      agent.state = node_defaults["state"].to float
    else:
      agent.state = state.random_state()

    if "edgeRate" in node_defaults:
      agent.edgeRate = node_defaults["edgeRate"].to float

    if "mutationRate" in node_defaults:
      agent.mutationRate = node_defaults["mutationRate"].to float

    # assign a bias
    agent.bias = 0.0
    if "bias" in node_defaults:
      agent.bias = node_defaults["bias"].to float

  for edge in g.edges():
    let x = edge[0].to int
    let y = edge[1].to int
    if x == y:
      continue
    state.agents[x].addEdge(state.agents[y], directed = g.is_directed().to bool)

proc normalize[T, U](d: var Table[T, U]) =
  let z = d.values().toseq().sum()
  for node, value in d.mpairs:
    value /= z

proc makeValueNetwork(state: var State, valueNetwork: PyObject, undirected = true) =
  # we initialize the value network
  state.valueNetwork = initTable[string, Table[string, float]]()
  state.config.p_roles = initTable[string, float]()
  # assign the roles based on the nodes
  for node in valueNetwork.nodes():
    discard state.config.p_roles.haskeyorput(&"{node}", 0.0)

  for x in valueNetwork.edges(data = true):
    let i = &"{x[0]}"
    let j = &"{x[1]}"
    let w = x[2].get("weight", 1.0).to(float)
    state.config.p_roles[i] += w
    state.config.p_roles[j] += w
    # undirected map
    if w == 0.0:
      continue
    if state.valueNetwork.haskeyorput(i, {j: w}.toTable()):
      discard state.valueNetwork[i].haskeyorput(j, w)
    if undirected:
      if state.valueNetwork.haskeyorput(j, {i: w}.toTable()):
        discard state.valueNetwork[j].haskeyorput(i, w)
  # normalize roles
  state.config.p_roles.normalize()
  # normalize the value network
  for key, value in state.valueNetwork.mpairs:
    value.normalize()

proc makeState*(config: Config, g, valueNetwork: PyObject): State =
  result = State(agents: @[], config: deepcopy(config), rng: initRand(config.seed))
  # add agents and connections
  result.makeValueNetwork(valueNetwork)
  result.makeNetwork(g)

proc setup*(base: Config, g, valueNetwork: PyObject): State =
  result = makeState(base, g, valueNetwork)
  result.rng = initRand(result.config.seed)

proc fermiUpdate*(delta, beta: float): float =
  result = 1.0 / (1.0 + exp(-(1 / beta) * delta))

proc energy(agent: Agent, interactions: seq[float], state: State): float =
  result =
    agent.parent.config.benefit * interactions.prod -
    agent.parent.config.cost * interactions[0]

proc getAvailableRoles*(agent: Agent): Table[string, seq[int]] =
  result = initTable[string, seq[int]]()
  for neighbor in agent.neighbors.keys():
    let role = agent.parent.agents[neighbor].role
    if result.haskeyorput(role, @[neighbor]):
      result[role].add neighbor

proc makeMutation(agent: Agent): Mutation {.inline.} =
  result = Mutation(
    id: agent.id, neighbors: agent.neighbors, state: agent.state, role: agent.role
  )

proc generateSnapshots*(t, n: int): seq[int] =
  return (0 ..< t).toseq()
  if n <= 0:
    return (0 ..< t).toseq()
  let first = (0.5 * n.float).int
  let second = (0.50 * n.float).int
  let m = (t - first).div(second)
  result = (1 ..< first).toseq().concat(countUp(first, t, m).toseq())

proc sample(agent: Agent, state: var State, order: int): float =
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
  for attempt in 1 ..< order:
    other = state.rng.sample(neighbors)
    if state.agents[other].role notin seen:
      interactions[attempt] = state.agents[other].state
      # attempt.inc
  result = agent.energy(interactions, state)

proc sampleNeighbor*(state: var State, agent: int): int {.inline.} =
  let agent = state.agents[agent]
  if agent.neighbors.len == 0 or state.rng.rand(1.0) < agent.mutationRate:
    var other = state.rng.sample(state.agents).id
    while state.agents[other].id == agent.id:
      other = state.rng.sample(state.agents).id
    return other

  # sample a random neighbor of neighbors
  if agent.neighbors.len > 0:
    result = state.rng.sample(agent.neighbors.keys().toseq())
    if state.agents[result].neighbors.len > 0:
      return state.rng.sample(state.agents[result].neighbors.keys().toseq())
  # default option:
  return agent.id

proc calculateCost*(state: State, agent: int, prior_cost: float): float {.inline.} =
  # compute the criminal cost proportional to its degree
  let z = 1.0 / (state.agents.len.float - 1.0)
  for neighbor in state.agents[agent].neighbors.keys():
    if state.agents[neighbor].state == 1.0:
      result += 1.0
  result = (z * result) * prior_cost

proc performEdgeAction(state: var State, agent, other: int) {.inline.} =
  # add or remove an edge depending on whether the edge
  # is already present
  if state.agents[agent].hasNeighbor(other):
    state.agents[agent].rmEdge(state.agents[other])
  else:
    state.agents[agent].addEdge(state.agents[other])

proc changeStrategy*(agent: var Agent) {.inline.} =
  agent.state = if agent.state == 1.0: 0.0 else: 1.0

proc acceptMutation(
    state: State, currents: seq[Mutation], mutations: var seq[Mutation]
) {.inline.} =
  for current in currents:
    mutations.add state.agents[current.id].makeMutation()

proc rejectMutation(state: var State, currents: seq[Mutation]) {.inline.} =
  for current in currents:
    state.agents[current.id].state = current.state
    state.agents[current.id].neighbors = current.neighbors
    state.agents[current.id].role = current.role

proc getPayoff*(state: var State, id: int, order = 3): float =
  if order < 1:
    raise (ref ValueError)(msg: "Order cannot be smaller than 1")
  var agent = state.agents[id]
  for sample in 0 ..< agent.nSamples:
    result += agent.sample(state, order)

proc step*(state: var State, agent: int, mutations: var seq[Mutation]) {.inline.} =
  var currents = @[state.agents[agent].makeMutation()] # store the current state
  var buffer = [0.0, 0.0] # change --> current, proposal
  let prior_cost = state.config.cost
  let prior_benefit = state.config.benefit

  # Cost is computed proportional to the criminal degree
  state.config.cost = calculateCost(state, agent, prior_cost)
  # compute the payoff in the current state
  let role = state.agents[agent].role
  let order = state.valueNetwork[role].len + 1
  buffer[0] = state.getPayoff(agent, order = order)
  if state.rng.rand(1.0) < state.agents[agent].edgeRate:
    let other = state.sampleNeighbor(agent)
    currents.add(state.agents[other].makeMutation())
    # add or remove an agent
    performEdgeAction(state, agent, other)
    # adj changes so we recompute the cost
    state.config.cost = calculateCost(state, agent, prior_cost)
    buffer[1] = state.getPayoff(agent)
  else:
    let prior = state.agents[agent].state
    changeStrategy(state.agents[agent])
    buffer[1] = state.getPayoff(agent)

  # check if we accept new state
  let delta = buffer[1] - buffer[0]
  # echo &"{z * delta=} {buffer=}"
  # sleep(100)
  let p = fermiUpdate(delta, state.config.beta)
  if state.rng.rand(1.0) < p:
    acceptMutation(state, currents, mutations)
  else:
    rejectMutation(state, currents)
  # reset parameters
  state.config.cost = prior_cost
  state.config.benefit = prior_benefit

proc simulate*(state: var State, t: int, n: int = 0): seq[seq[Mutation]] =
  # keep diffs for copy
  var agents = (0 ..< state.agents.len).toseq

  var snapshots = generateSnapshots(t, n)
  result = newSeq[newSeq[Mutation]()](snapshots.len)
  # result = newSeq[newSeq[Mutation]()](t)
  # NOTE: mutations are now stored dense in its adjacency structure
  # since we are having additions and removals, we need to keep track
  # of both which could be remapped to positive or negative indices
  # to save on memory.

  var mutations = state.agents.mapIt(it.makeMutation)
  var snap = 0

  for ti in 0 ..< t:
    if ti == snapshots[snap]:
      result[snap] = mutations
      snap.inc
    mutations = @[]
    state.rng.shuffle(agents)
    # let agent = state.rng.sample(agents)
    # state.step(state.rng.sample(agents), mutations)
    for agent in agents:
      #NOTE: will add to mutations if new state is accepted
      state.step(agent, mutations)

proc makeBuffer(
    n: int, state: var State, agents: var seq[int], mutations: var seq[Mutation]
): seq[float] =
  result = newSeqWith[float](n, 1.0)
  let z = 1 / n.float
  # fill the buffer
  for idx in 0 ..< result.len:
    result[idx] =
      state.agents.foldl(a + b.neighbors.len, state.agents[0].neighbors.len).float * z
    # perform a step
    agents.shuffle()
    mutations = @[] # empty mutations to prevent blow up
    for agent in agents:
      step(state, agent, mutations)

proc findEquilibrium(
    state: var State,
    agents: var seq[int],
    mutations: var seq[Mutation],
    threshold: float,
) =
  var buffer = makeBuffer(100, state, agents, mutations)
  let nbuf = buffer.len
  let z = 1 / nbuf.float
  var zz = state.agents.len.float
  zz = 1 / (zz * (zz - 1) / 2.0)
  var idx = 0
  # equilibrate
  proc mse(buffer: seq[float]): float =
    for idx, other in buffer[1 ..^ 1]:
      result += (other - buffer[idx]) ^ 2
    result *= 1 / (buffer.len.float)

  idx = 0
  while mse(buffer) > threshold:
    buffer[idx] =
      state.agents.foldl(a + b.neighbors.len, state.agents[0].neighbors.len).float * zz
    idx = (idx + 1).mod(nbuf)
    # perform a step
    agents.shuffle()
    mutations = @[] # empty mutations to prevent blow up
    for agent in agents:
      step(state, agent, mutations)

proc simulateInEquilibrium*(
    state: var State, n = 0, threshold = 1e-2, mutationAfter = 1.0
): seq[seq[Mutation]] =
  assert n >= 1
  var agents = (0 ..< state.agents.len).toseq()
  # NOTE: the first index is the startin state and should contain all the agents
  result = newSeq[newSeq[Mutation]()](n)
  var mutations = state.agents.mapIt(it.makeMutation)
  result[0] = mutations

  # we equilibrate in the edge change
  var z = state.agents.len.float
  z = z * (z - 1) / 2

  state.findEquilibrium(agents, mutations, threshold)
  for agent in state.agents:
    agent.mutationRate = mutationAfter

  # sample from the equilibrium
  for sample in 1 ..< n:
    result[sample] = mutations
    # perform a step
    agents.shuffle()
    mutations = @[] # empty mutations to prevent blow up
    for agent in agents:
      step(state, agent, mutations)

proc `echo`*(config: Config) =
  echo '-'.repeat(16), " Parameters ", '-'.repeat(16)
  for key, value in config[].fieldPairs():
    echo key, ": ", value
