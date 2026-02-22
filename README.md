# Arcade prediction benchmark
A prediction benchmark for pre-trained policies on Atari2600 games The policies are taken from pre-trained rainbow DQN agent from the Chainer RL model zoo [1].

A link to the policies is [here](https://drive.google.com/file/d/1EYHOLrAtBiVHJKOYK3VNvoxDFWF7o71S/view?usp=share_link).

For each game, there are four epsilon-greedy policies with different epsilons. The epsilon values are 0.0, 0.05, 0.1, and 0.2. Epsilon 0.1 is a reasonable default. 

## Run a policy
Use `test.py` to replay a policy locally.

Example using `Pong_eps_0_1`:
```bash
python test.py policies/rainbow_Pong_eps_0.1.json
```

This runs the `Pong_eps_0_1` policy from the `policies/` folder and prints the episode returns.

## Export traces to a dataset (`policy_to_dataset.py`)

Replay one or more policy JSONs and save observations, rewards, terminals, actions, and discounted returns to a single `.npz` file. Requires `ale-py`, `numpy`, and Atari ROMs (set `ALE_ROMS_DIR` if ROMs are not in the package directory).

**Usage:**
```bash
# All steps from one or more policy files (returns use gamma=0.99 by default)
python policy_to_dataset.py -o dataset.npz policies/rainbow_Pong_eps_0.1.json

# Cap at 200 steps per file; each file must have at least 200 + 5000 steps (buffer for return computation)
python policy_to_dataset.py --steps 200 --output pong_breakout.npz policies/rainbow_Pong_eps_0.1.json policies/rainbow_Breakout_eps_0.1.json

# Custom discount factor for returns (default 0.99)
python policy_to_dataset.py --gamma 0.95 -o out.npz policies/rainbow_Pong_eps_0.1.json
```

**Options:**
- `--output`, `-o`: Output `.npz` path (default: `dataset.npz`).
- `--steps` N: Use at most N steps per policy file; each file must have at least N + 5000 steps when set.
- `--gamma` G: Discount factor for return computation (default: 0.99). Returns are discounted sums of *future* rewards (from the next step onward).

The `.npz` contains: `observations`, `rewards`, `terminals`, `actions`, `returns`.

## Visualize a dataset (`visualize_dataset.py`)

Render a dataset `.npz` as a video with reward, return, and terminal overlaid on each frame (green text).

**Usage:**
```bash
# Full dataset, output to <dataset_stem>_visualization.mp4
python visualize_dataset.py dataset.npz

# Specify output path and FPS
python visualize_dataset.py dataset.npz -o video.mp4 --fps 30

# Limit to first 10 seconds of video
python visualize_dataset.py dataset.npz --cap-length 10 -o short.mp4
```

**Options:**
- `--output`, `-o`: Output video path (default: `<dataset_stem>_visualization.mp4`).
- `--fps`: Frames per second (default: 60).
- `--cap-length` N: Only write the first N seconds of the dataset.

Requires `imageio` and `Pillow` (see `requirements.txt`).



