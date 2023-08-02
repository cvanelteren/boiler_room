import std/[sequtils, random, math, strutils,
            strformat, strutils, tables, sequtils, terminal]
import nimpy

type Agent* = ref object
  id*: int
  state*: float
  role*: string
  neighbors*: seq[ptr Agent]
  trust*: Table[int, float]
  bias*: float

type Config* = ref object of RootObj
  states*: seq[float]
  roles*: seq[string]
  alpha*, beta*, benefit*, cost*: float


type State* = ref object of Config
  agents*: seq[Agent]
  rng*: Rand

# proc `echo`*(agent: Agent) =
  # echo &"ID: {agent.id}\n State: {agent.state}\nnumber of neighbors: {agent.neighbors.len}"

proc makeSimulation*(config: Config): State =
  result = State(states: config.states,
                 roles: config.roles,
                 benefit: config.benefit,
                 cost: config.cost,
                 beta: config.beta,
                 alpha: config.alpha,
                 agents: @[],
  )

proc makeAgent*(id: int, state: State): Agent =
  result = Agent(id: id, neighbors: @[],
                 role: state.roles.sample,
                 state: state.states.sample,
                 bias: 0.0,
                 trust: initTable[int, float]())

proc makeNetwork*(state: var State, g: PyObject) =
  # add agents
  for node in g.nodes():
    state.agents.add makeAgent(node.to(int), state)
    let tmp = g.nodes[node].to Table[string, PyObject]
    for key, value in tmp:
      if key == "role":
        state.agents[^1].role = value.to string
        assert state.agents[^1].role == value.to string
      elif key == "state":
        state.agents[^1].state = value.to float
      elif key == "bias":
        state.agents[^1].bias = value.to float

  for edge in g.edges():
    let x = edge[0].to int
    let y = edge[1].to int
    if x == y:
      continue
    state.agents[x].neighbors.add state.agents[y].addr
    state.agents[x].trust[state.agents[y].id] = 1.0
    if not g.is_directed().to bool:
        state.agents[y].neighbors.add state.agents[x].addr
        state.agents[y].trust[state.agents[x].id] = 1.0


# not a real cdf but sample can deal with counts
proc getTrustCDF*(agent: Agent): seq[float] = agent.neighbors.mapIt(agent.trust[it.id]).cumsummed

proc energy(agent: Agent, interactions: seq[float], state: State): float =
  # assume that 0 index is the current agent energy
  result = state.benefit * interactions.prod - state.cost * interactions[0] + interactions[0] * agent.bias

proc fermiUpdate*(delta, beta: float): float {.exportpy.} =
  result = 1.0 / (1.0 + exp(-beta * delta))

proc updateTrust*(agent: Agent, alpha: float) =
  if alpha > 0:
    # update trust only if agent state is the same as the neighbor
    for neighbor in agent.neighbors:
      if neighbor.state == agent.state:
        agent.trust[neighbor.id] *= (1 + alpha)
      else:
        agent.trust[neighbor.id] *= (1 - alpha)


proc update(state: var State, id: int, order = 3) =
  if order < 1:
    raise (ref ValueError)(msg: "Order cannot be smaller than 1")
  # fermi update
  let agent = state.agents[id]
  var
    roles: seq[string] = @[agent.role]
    ids: seq[int] = @[agent.id]
    interactions: seq[float] = newSeqWith(order, 0.0)
    idx = 0

  interactions[0] = agent.state

  # shuffle iteration
  # TODO: bin the roles of the neighbors in to group

  let cdf = agent.getTrustCDF

  while roles.len < (order) and idx < agent.neighbors.len:
    let other = state.rng.sample(agent.neighbors, cdf = cdf)
    if other.role notin roles and other.id notin ids:
      interactions[roles.len] = other.state
      ids.add other.id
      roles.add other.role
    idx.inc

  var currentEnergy = agent.energy(interactions, state)

  # assume new role and compute energy difference
  proc filter(this, other: float, n: int): float =
    result = 0.0
    if this != other:
      result = 1/(n - 1)

  let ps = state.states.mapIt(
    filter(it, agent.state, state.states.len))

  interactions[0] = state.rng.sample(state.states, ps.cumsummed)
  # echo state.states.sample(ps.cumsummed), ps
  assert interactions[0] != agent.state
  var flipEnergy = agent.energy(interactions, state)

  let delta = flipEnergy - currentEnergy
  let p = fermiUpdate(delta, state.beta)
  # echo &"{p=} {delta=} {currentEnergy=} {flipEnergy=}"
  if rand(1.0) < p:
    agent.state = interactions[0]
  agent.updateTrust(state.alpha)

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

    # let agent = rand(state.agents.len - 1)
    # state.update(agent)

    state.rng.shuffle(agents)
    for agent in agents:
      state.update(agent)


proc `echo`*(config: Config) =
  echo '-'.repeat(16), " Parameters ", '-'.repeat(16)
  for key, value in config[].fieldPairs():
    echo key, ": ", value

proc run*(parameters: PyObject): PyObject {.exportpy.} =
  # @parameters should contain a dict
  # echo "Parsing input"
  var config = Config(states: @[0.0, 1.0],
                     roles: "Production Distribution Management".split,
                     beta: 1.0,
                     cost: 1.0,
                     alpha: 1.0,
                     benefit: 1.0)

  var parameters = parameters.to Table[string, PyObject]
  # assert not parameters["g"].isnil

  for key, value in config[].fieldPairs:
    if key in parameters:
      value = parameters[key].to(type value)

  var state = makeSimulation(config)
  var t: int = 1000
  if "t" in parameters:
    t = parameters["t"].to int

  # echo config
  # echo &"Running simulation {t=}"


  result = makeExport(state.simulate(t))
