import unittest
from boiler_room.utils import read, Config

suite "Config parser":
  var config: Config
  test "Parsing general":
    config = "./tests/default.toml".read()
    check config.states == @[0.0]
    check config.roles == @["A"]
    check config.z == 1
    check config.t == 1
    check config.p_states == @[@[0.0]]
    check config.p_roles == @[1.0]
    check config.p_roles.len == config.roles.len
    check config.states.len == config.p_roles.len
    check config.beta == 1.0
    check config.benefit == 1.0
    check config.cost == 1.0
    # check config.p_state.all((it: seq[float]) => bool = it.len == config.states.len)

  test "Parsing override":
    config = "./tests/default.toml".read(target = "test ratio input")
    check config.states == @[0.0]
    check config.roles == @["A"] # this is changed
    check config.z == 1
    check config.t == 1
    check config.p_states == @[@[0.0]]
    check config.p_roles == @[1.0]
    check config.p_roles.len == config.roles.len
    check config.states.len == config.p_roles.len
    check config.beta == 1.0

    check config.benefit == 2.0
    check config.cost == 1.0

  test "parsing fraction":
    config = "./tests/default.toml".read(target = "test fraction")
    check config.p_roles == @[0.33333]
