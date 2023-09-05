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

type State* = object of Config
  agents*: seq[Agent]
  rng*: Rand

proc get*(agents: seq[ptr Agent]): seq[int] =
  result = agents.mapIt(it.id)

proc makeAgent*(id: int, state: State): Agent =
  result = Agent(id: id,
                 neighbors: initTable[string, seq[ptr Agent]](),
                 role: state.roles.sample,
                 state: state.states.sample,
                 bias: 0.0,
                 n_samples: 1
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
    let agent = makeAgent(node.to(int), state)

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

    # assign an explore rate
    agent.n_samples = 1
    if "n_samples" in node_defaults:
      agent.n_samples = node_defaults["n_samples"].to int
    state.agents.add agent

  for edge in g.edges():
    let x = edge[0].to int
    let y = edge[1].to int
    if x == y:
      continue
    state.agents[x].add_neighbor(state.agents[y],
                                 directed = g.is_directed().to bool)

proc makeState*(config: Config, g: PyObject): State =
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
                 agents: @[],
  )
  # add agents and connections
  result.makeNetwork(g)

proc energy(agent: Agent, interactions: seq[float], state: State): float =
  # assume that 0 index is the current agent energy
  result = state.benefit * interactions.prod -
    state.cost * interactions[0] +
    interactions[0] * agent.bias

proc fermiUpdate*(delta, beta: float): float {.exportpy.} =
  result = 1.0 / (1.0 + exp(-beta * delta))




# how many time to sample... sample until you get a hit?
proc sample(agent: Agent, state: var State, ids: var seq[int], order: int): float =
  # fermi update
  var
    roles: seq[string] = @[agent.role]
    interactions: seq[float] = newSeqWith(order, 0.0)
    idx = 0

  interactions[0] = agent.state
  # let cdf = agent.getTrustCDF
  # echo agent.neighbors.len
  # TODO: sample non-criminal first
  for role, neighbors in agent.neighbors:
    if role != agent.role:
      let tmp = state.rng.sample(neighbors)

  var currentEnergy = agent.energy(interactions, state)
  proc filter(this, other: float, n: int): float =
    result = 0.0
    if this != other:
      result = 1/(n - 1)

  let ps = state.states.mapIt(
    filter(it, agent.state, state.states.len))

  interactions[0] = state.rng.sample(state.states, ps.cumsummed)
  assert interactions[0] != agent.state
  var flipEnergy = agent.energy(interactions, state)
  result = flipEnergy - currentEnergy


proc update(state: var State, id: int, order = 3) =
  if order < 1:
    raise (ref ValueError)(msg: "Order cannot be smaller than 1")
  let agent = state.agents[id]
  var delta = 0.0
  var ids = @[agent.id]
  let z = 1/((float) agent.n_samples)
  for sample in 0..<agent.n_samples:
    delta += agent.sample(state, ids, order)
  let p = fermiUpdate(delta * z, state.beta)
  if state.rng.rand(1.0) < p:
    if agent.state == 1.0:
      agent.state = 0.0
    else:
      agent.state = 1.0
  # agent.updateTrust(ids, state.alpha)

proc makeExport*(states: seq[State]): PyObject=
  let pd = pyImport("pandas")
  result = pd.DataFrame()
  for idx, state in states:
    let data =  @[(state: state.agents.mapIt(it.state),
                 roles: state.agents.mapIt(it.role),
                 beta: state.beta )]
    let tmp = pd.DataFrame(data = data, columns = "state roles g beta".split)
    result = pd.concat((result, tmp))

proc simulate*(state: var State, t: int): seq[State] =
  var agents = (0..<state.agents.len).toSeq
  # keep diffs for copy
  for ti in 0..<t:
    result.add deepcopy(state)

    state.rng.shuffle(agents)
    for agent in agents:
      state.update(agent)

proc `echo`*(config: Config) =
  echo '-'.repeat(16), " Parameters ", '-'.repeat(16)
  for key, value in config.fieldPairs():
    echo key, ": ", value
