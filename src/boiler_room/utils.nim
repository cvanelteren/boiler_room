import sequtils, parsetoml, strutils, strformat
from agent import Config, Mutation, DataPoint, State

proc to*(mutations: seq[seq[Mutation]], T: typedesc): DataPoint =
  result.states = newSeqWith(mutations.len, newSeq[float]())
  result.adj = newSeqWith(mutations.len, initTable[int, seq[int]]())
  for agent in mutations[0]:
    result.states[0].add agent.state
    result.adj[0][agent.id] = agent.neighbors.keys().toseq()

  # start from 1 but the index here starts at 0
  for kdx, mutation in mutations[1 ..^ 1]:
    result.states[kdx + 1] = result.states[kdx]
    result.adj[kdx + 1] = result.adj[kdx]
    for agent in mutation:
      result.states[kdx + 1][agent.id] = agent.state
      result.adj[kdx + 1][agent.id] = agent.neighbors.keys().toseq()

proc create_data_name(
    base: string, config: Config, ext = ".json", additional = ""
): string =
  result = [
    &"{config.p_states=}",
    &"{config.trial=}",
    &"{config.beta=}",
    &"{config.cost=}",
    &"{config.z=}",
  ].join("_")
  result.add additional
  result = result.replace("config.", "")
  result = [base, "/", result, ext].join()

proc create_graph_filename(
    base: string, config: Config, ext = ".graph", additional = ""
): string =
  result = create_data_name(base, config, ext = ".graph", additional = additional)

proc readParams*(fp: string, target: string = "general"): Config =
  # TODO: very uggly replace this with more readable code
  let tmp = parsetoml.parseFile(fp).getTable
  result = Config()

  # read p_state
  var s = tmp["general"]["p_states"].getElems.mapIt(it.getFloat)
  if "p_states" in tmp[target]:
    s = tmp[target]["p_states"].getElems.mapIt(it.getFloat)
  result.p_states = initTable[float, float]()
  for idx, si in s:
    result.p_states[idx.float] = si

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

  result.step = 1
  if "N" in tmp[target]:
    result.step = tmp[target]["N"].getInt()
