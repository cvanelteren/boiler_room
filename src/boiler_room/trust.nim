#TODO: currently removed since I am narrowing down a bug. Reimplement later with a better
#strategy
#
#

# proc add_trust(state: var State) =
#   state.trust = newSeqWith(state.agents.len,
#                         newSeqWith(state.agents.len, 1.0))
#   for agent in state.agents:
#     agent.trust = state.trust.addr

# proc getTrustCDF*(agent: Agent): seq[float] =
#   for neighbor in agent.neighbors:
#     result.add agent.trust[agent.id][neighbor.id]

# proc updateTrust*(agent: Agent, ids: seq[Agent], alpha: float) =
#   var update: float
#   if alpha > 0:
#     # update trust only if agent state is the same as the neighbor
#     for neighbor in ids:
#       if neighbor.id == agent.id:
#         continue
#       if neighbor.state == agent.state:
#         update = 1 + alpha
#       else:
#         update = 1 - alpha
#       agent.trust[agent.id][neighbor.id] *= update
#       agent.trust[neighbor.id][agent.id] *= update
