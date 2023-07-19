import std/[sequtils, random, math, strutils, strformat, strutils, tables, sequtils, terminal]
import nimpy

# randomize()
type Agent* = ref object
  id: int
  state: float
  role: string
  neighbors: seq[Agent]
  bias: float

type Config* = ref object of RootObj
  g*: PyObject
  states*: seq[float]
  roles*: seq[string]
  beta*, benefit*, cost*: float

type State* = object of Config
  agents: seq[Agent]

proc makeAgent*(id: int, state: State): Agent =
  result = Agent(id: id, neighbors: @[],
                 role: state.roles.sample,
                 state: state.states.sample,
                 bias: 0.0)

proc makeSimulation*(config: Config): State =
  result = State(states: config.states,
                 roles: config.roles,
                 benefit: config.benefit,
                 cost: config.cost,
                 beta: config.beta,
                 g: config.g,
                 agents: @[],
  )
  # add agents
  for node in config.g.nodes():
    result.agents.add makeAgent(node.to(int), result)
    let tmp = config.g.nodes[node].to Table[string, PyObject]
    for key, value in tmp:
      if key == "role":
        result.agents[^1].role = value.to string
        assert result.agents[^1].role == value.to string
      elif key == "state":
        result.agents[^1].state = value.to float
      elif key == "bias":
        result.agents[^1].bias = value.to float

  for edge in config.g.edges():
    let x = edge[0].to int
    let y = edge[1].to int
    if x == y:
      continue
    result.agents[x].neighbors.add result.agents[y]
    if not config.g.is_directed().to bool:
        result.agents[y].neighbors.add result.agents[x]


proc energy(agent: Agent, interactions: seq[float], state: State): float =
  # assume that 0 index is the current agent energy
  result = state.benefit * interactions.prod - state.cost * interactions[0] + interactions[0] * agent.bias

proc fermiUpdate*(delta, beta: float): float =
  result = 1.0 / (1.0 + exp(-beta * delta))

proc update(state: var State, id: int, order = 3) =
  if order < 1:
    raise (ref ValueError)(msg: "Order cannot be smaller than 1")
  # fermi update
  let agent = state.agents[id]
  var
    roles: seq[string] = @[agent.role]
    interactions: seq[float] = newSeqWith(order, 0.0)
    idx = 0

  interactions[0] = agent.state

  # shuffle iteration
  # TODO: bin the roles of the neighbors in to group
  agent.neighbors.shuffle()

  while roles.len < (order) and idx < agent.neighbors.len:
    let other = agent.neighbors[idx]
    if other.role notin roles:
      interactions[roles.len] = other.state
      roles.add other.role
    idx.inc

  var currentEnergy = agent.energy(interactions, state)

  # assume new role and compute energy difference
  proc filter(this, other: float, n: int): float =
    result = 0.0
    if this != other:
      result = 1/(n - 1)

  let ps = state.states.mapIt(
    filter(it, agent.state, state.states.len)).cumsummed

  interactions[0] = state.states.sample(ps)
  assert interactions[0] != agent.state
  var flipEnergy = agent.energy(interactions, state)

  let delta = flipEnergy - currentEnergy
  let p = fermiUpdate(delta, state.beta)
  # echo &"{p=} {delta=} {currentEnergy=} {flipEnergy=}"
  if rand(1.0) < p:
    agent.state = interactions[0]

proc makeExport(states: seq[State]): PyObject=
  let pd = pyImport("pandas")
  result = pd.DataFrame()
  for idx, state in states:
    let data =  @[(state: state.agents.mapIt(it.state),
                 roles: state.agents.mapIt(it.role),
                 g: state.g,
                 beta: state.beta )]
    let tmp = pd.DataFrame(data = data, columns = "state roles g beta".split)
    result = pd.concat((result, tmp))



proc simulate*(state: var State, t: int): seq[State] =
  var agents = (0..<state.agents.len).toSeq

  for ti in 0..<t:
    let i = ((ti / t) * 100).toInt
    # stdout.styledWriteLine(fgRed, "0% ", fgWhite, '#'.repeat i, if i > 50: fgGreen else: fgYellow, "\t", $i , "%")
    # cursorUp 1
    # eraseLine()
    agents.shuffle()
    result.add deepcopy(state)
    # let agent = rand(state.agents.len - 1)
    for agent in agents:
      state.update(agent)

    # for agent in agents:
      # state.update(agent)
      # let agent = rand(state.agents.len - 1)

proc `echo`*(config: Config) =
  echo '-'.repeat(16), " Parameters ", '-'.repeat(16)
  for key, value in config[].fieldPairs():
    echo key, ": ", value

proc run*(parameters: PyObject): PyObject {.exportpy.} =
  # @parameters should contain a dict
  echo "Parsing input"
  var config = Config(states: @[0.0, 1.0],
                     roles: "Production Distribution Management".split,
                     beta: 1.0,
                     cost: 1.0,
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

  echo config
  echo &"Running simulation {t=}"
  result = makeExport(state.simulate(t))
