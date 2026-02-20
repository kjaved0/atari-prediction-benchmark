# Arcade prediction benchmark
For benchmark policy evaluation on Atari. The policies are taken from pre-trained rainbow DQN agent from the Chiner RL model zoo [1].

A small test policy for Pong with sample code is available in test.py. 

Link to full policies: [Google drive](https://drive.google.com/file/d/1ouaERgS5rt6KsX77JpULe8OcYehtsYXt/view?usp=sharing).

## Run a policy
Use `test.py` to replay a policy locally.

Example using `Pong_eps_0_1`:
```bash
python test.py policies/rainbow_Pong_eps_0.1.json
```

This runs the `Pong_eps_0_1` policy from the `policies/` folder and prints the episode returns.


You can cite the benchmark as:
``` bash
@article{javed2023scalable,
  title={Scalable real-time recurrent learning using columnar-constructive networks},
  author={Javed, Khurram and Shah, Haseeb and Sutton, Richard S and White, Martha},
  journal={Journal of Machine Learning Research},
  volume={24},
  pages={1--34},
  year={2023}
}
```


[1] Fujita, Yasuhiro, et al. "Chainerrl: A deep reinforcement learning library." The Journal of Machine Learning Research 22.1 (2021): 3557-3570.


