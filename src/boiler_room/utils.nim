import sequtils, parsetoml
import nimpy

type Config* = object of RootObj
  states*: seq[float]
  roles*: seq[string]

  alpha*, beta*, benefit*, cost*: float
  mu*: float # deprecated

  rewire*: float
  assortativity*: float
  depth*:int
  g*: PyObject

  n_samples*, t*, seed*, z*: int
  trial*, n_trials*: int

  p_states*: seq[seq[float]]
  p_roles* : seq[float]

proc read*(fp: string, target: string = "general"): Config =
  # TODO: very uggly replace this with more readable code
  let tmp = parsetoml.parseFile(fp).getTable
  result = Config()

  # read states
  result.states = tmp["general"]["states"].getElems.mapIt(it.getFloat)
  if "states" in tmp[target]:
    result.states = tmp[target]["states"].getElems.mapIt(it.getFloat)

  # read roles
  result.roles = tmp["general"]["roles"].getElems.mapIt(it.getStr)
  if "roles" in tmp[target]:
    result.roles = tmp[target]["roles"].getElems.mapIt(it.getStr)

  # read p_state
  result.p_states = tmp["general"]["p_states"].getElems.mapIt(it.getElems.mapIt(it.getFloat))
  if "p_states" in tmp[target]:
    result.p_states = tmp[target]["p_states"].getElems.mapIt(it.getElems.mapIt(it.getFloat))

  assert result.p_states.len == result.roles.len

  # read p_roles
  result.p_roles = tmp["general"]["p_roles"].getElems.mapIt(it.getFloat)
  if "p_roles" in tmp[target]:
    result.p_roles = tmp[target]["p_roles"].getElems.mapIt(it.getFloat)


  # load constants
  result.z = tmp["general"]["z"].getInt
  if "z" in tmp[target]:
    result.z = tmp[target]["z"].getInt

  result.t = tmp["general"]["t"].getInt
  if "t" in tmp[target]:
    result.t = tmp[target]["t"].getInt

  result.beta = tmp["general"]["beta"].getFloat
  if "beta" in tmp[target]:
    result.beta = tmp[target]["beta"].getFloat

  result.seed = tmp["general"]["seed"].getInt
  if "seed" in tmp[target]:
    result.seed = tmp[target]["seed"].getInt

  # check for ratio
  if "ratio" in tmp[target]:
    result.benefit = tmp[target]["ratio"].getFloat
    result.cost = 1
  elif "ratio" in tmp["general"]:
    result.benefit = tmp["general"]["ratio"].getFloat
    result.cost = 1
  else:
    result.benefit = tmp["general"]["benefit"].getFloat
    if "benefit" in tmp[target]:
      result.benefit = tmp[target]["benefit"].getFloat

    result.cost = tmp["general"]["cost"].getFloat
    if "cost" in tmp[target]:
      result.cost = tmp[target]["cost"].getFloat

  result.n_trials = tmp["general"]["n_trials"].getInt()
  if "n_trials" in tmp[target]:
    result.n_trials = tmp[target]["n_trials"].getInt()

  result.n_samples = tmp["general"]["n_samples"].getInt()
  if "n_samples" in tmp[target]:
    result.n_samples = tmp[target]["n_samples"].getInt()


  result.mu = 0.0
  if "mu" in tmp[target]:
    result.mu = tmp[target]["mu"].getFloat()
