import base64, json, pickle, sys
from ale_py import ALEInterface, roms

trace = json.load(open(sys.argv[1]))
rom = trace["env"].split("-v")[0].replace("NoFrameskip", "").replace("Deterministic", "")

ale = ALEInterface()
ale.setInt("random_seed", int(trace["test_seed"]))
ale.setFloat("repeat_action_probability", 0.0)
ale.setInt("frame_skip", 1)
ale.loadROM(getattr(roms, rom))
ale.reset_game()
ale.restoreState(pickle.loads(base64.b64decode(trace["ale_state"])))

actions = list(ale.getMinimalActionSet())
ep_return = 0.0
episode_returns = []

for c in trace["sequence_of_actions"]:
    if c == "R":
        episode_returns.append(ep_return)
        print(episode_returns)
        ep_return = 0.0
        ale.reset_game()
    else:
        ep_return += ale.act(actions[ord(c) - ord("a")])

if ep_return != 0:
    episode_returns.append(ep_return)

print("episode_returns =", episode_returns)
print("expected_returns =", trace.get("episode_returns"))
