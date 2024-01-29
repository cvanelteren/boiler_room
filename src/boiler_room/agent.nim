import sequtils, random, math, strutils,
            strformat, strutils, tables, sequtils, terminal, options
import utils
import nimpy

type Agent* = ref object
  id*: int
  state*: float
  role*: string
  neighbors*: Table[string, seq[ptr Agent]]
  bias*: float
  n_samples*: int
  edgeRate*: float
  addRate*: float

type Mutation* = object
    id*: int
    state*: float
    neighbors*: seq[int]
    role*: string


type State* = object of Config
  agents*: seq[Agent]
  rng*: Rand
  p*: Table[string, Table[float, float]]

proc get*(agents: seq[ref Agent]): seq[int] =
  result = agents.mapIt(it.id)

proc find(neighbors: seq[ptr Agent], id: int): int =
  result =  -1
  for idx, neighbor in neighbors:
    if neighbor.id == id:
      return idx


proc makeAgent*(id: int, state: State): Agent =
  result = Agent(id: id,
                 neighbors: initTable[string, seq[ptr Agent]](),
                 role: state.roles.sample,
                 state: state.states.sample,
                 bias: 0.0,
                 n_samples:  state.n_samples,
                 edgeRate: 0.0,
                 addRate: 0.0
                 )

proc addEdge*(this: var Agent, other: var Agent, directed = false) =
  if this == other:
    return
  if this.neighbors.hasKeyOrPut(other.role, @[other.addr]):
    let idx = this.neighbors[other.role].find(other.id)
    if idx == -1:
      this.neighbors[other.role].add other.addr
  if not directed:
    if other.neighbors.hasKeyOrPut(this.role, @[this.addr]):
      let idx = other.neighbors[this.role].find(this.id)
      if idx == -1:
        other.neighbors[this.role].add this.addr

proc rmEdge*(this, other: var Agent, directed = false) =
  # remove edge
  if this.neighbors.hasKey(other.role):
    if this.neighbors[other.role].len > 1:
      let idx = this.neighbors[other.role].find(other.id)
      if idx != -1:
        this.neighbors[other.role].del idx

  if not directed:
    if other.neighbors.hasKey(this.role):
      if other.neighbors[this.role].len > 1:
        let idx =  other.neighbors[this.role].find(this.id)
        if idx != -1:
          other.neighbors[this.role].del idx

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

    if "addRate" in node_defaults:
      agent.addRate = node_defaults["addRate"].to float

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
  )
  # add agents and connections
  result.makeNetwork(config.g)



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
            ): float =
  # fermi update
  var
    interactions: seq[float] = newSeqWith(order, 0.0)
    idx = 1 # skip first index; only interested in neighborstates
  interactions[0] = agent.state
  # Create distribution of surrounding agent roles and states
  for role, neighbors in agent.neighbors:
    if role != agent.role:
      if neighbors.len > 0:
        interactions[idx] = state.rng.sample(neighbors).state
      idx.inc
  result = agent.energy(interactions, state)

proc sample(neighbors: Table[string, seq[ptr Agent]],
            state: var State): ptr Agent =
  var role = ""
  role = state.rng.sample(neighbors.keys().toseq())
  return state.rng.sample(neighbors[role])

proc update(state: var State, id: int, order = 3): float =
  if order < 1:
    raise (ref ValueError)(msg: "Order cannot be smaller than 1")
  var agent = state.agents[id]
  for sample in 0..<agent.n_samples:
    result += agent.sample(state, order)

proc makeMutation(agent: Agent): Mutation =
  result = Mutation(
      id: agent.id,
      neighbors: @[],
      state: agent.state,
      role: agent.role
    )
  for role, neighbors in agent.neighbors:
    for neighbor in neighbors:
      result.neighbors.add neighbor.id


proc simulate*(state: var State, t: int): seq[seq[Mutation]] =
  # keep diffs for copy
  var agents = (0..<state.agents.len).toseq
  result = newSeq[State](t)

  var buffer = newSeq[float](state.states.len)
  var other: int
  var added = false
  var states: seq[float]
  let order = state.roles.len
  var mutations = state.agents.mapIt(it.makeMutation)
  for ti in 0..<t:
    agents.shuffle()
    result.add mutations
    mutations = @[]
    for agent in agents:
      # perform a neighbor update..
      other = -1
      added = false
      states = @[state.agents[agent].state]
      if state.rng.rand(1.0) < state.agents[agent].edgeRate:
        # get the current energy
        buffer[0] = state.update(agent, order)

        # we add an edge
        if state.rng.rand(1.0) < state.agents[agent].addRate:
          added = true
          var pool = (0..<state.agents.len).toseq()
          for role, neighbors in state.agents[agent].neighbors:
            for n in neighbors:
              let idx = pool.find(n.id)
              if idx != -1:
                pool.del idx

          pool.del pool.find agent
          if pool.len > 0:
            # we cannot add if we do not have any neighbors that we are not
            # already connected to
            other = state.rng.sample(pool)
            state.agents[agent].addEdge(state.agents[other])
          else:
            continue
        else:
          # we can remove an existing enighbor
          if state.agents[agent].neighbors.len > 0:
            other = (state.agents[agent].neighbors.sample(state)).id
            state.agents[agent].rmEdge(state.agents[other])
          # TODO: check this
          else:
            continue
        buffer[1] = state.update(agent, order)
      # ..or a state update
      else:
        for newState in state.states:
          if newState != state.agents[agent].state:
            states.add newState

        for idx, newState in states:
          state.agents[agent].state = newState
          buffer[idx] = state.update(agent, order)


      # update the agent
      let delta = (buffer[1] - buffer[0])#/(state.agents[agent].n_samples.float)
      # accept the state
      if state.rng.rand(1.0) < fermiUpdate(delta, state.beta):
        # only update if we are considering updating the state
        if other == -1:
          state.agents[agent].state = states[1]
        # add the mutation
        mutations.add state.agents[agent].makeMutation()
      else:
        if other != -1:
          if added:
            state.agents[agent].rmEdge(state.agents[other])
          else:
            state.agents[agent].addEdge(state.agents[other])
        else:
          state.agents[agent].state = states[0]


proc `echo`*(config: Config) =
  echo '-'.repeat(16), " Parameters ", '-'.repeat(16)
  for key, value in config.fieldPairs():
    echo key, ": ", value
