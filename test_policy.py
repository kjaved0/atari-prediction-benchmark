import base64, json, pickle, re, sys
from ale_py import ALEInterface, roms

trace = json.load(open(sys.argv[1]))
rom_name = trace["env"].split("-v")[0].replace("NoFrameskip", "").replace("Deterministic", "")
# Convert CamelCase to snake_case for ale-py ROM IDs (e.g. BeamRider -> beam_rider, Breakout -> breakout)
rom_id = re.sub(r"(?<!^)(?=[A-Z])", "_", rom_name).lower()
rom_path = roms.get_rom_path(rom_id)

ale = ALEInterface()
ale.setInt("random_seed", int(trace["test_seed"]))
ale.setFloat("repeat_action_probability", 0.0)
ale.setInt("frame_skip", 1)
ale.loadROM(rom_path)
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
