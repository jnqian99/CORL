from myenv_gru import MyEnv
import random
import numpy as np

chkpt_dir = '/tmp/chkpt/myenv_gru/halfcheetah/medium_v2'

def main():
    env = MyEnv()
    env.load_from_checkpoint(chkpt_dir)
    # create a random float array of size 17
    obs = np.random.rand(env.get_state_size())
    env.reset(obs)
    print("init obs:", obs)
    for i in range(10):
        action = np.random.rand(env.get_action_size())
        obs, reward, bdone, _ = env.step(action)
        print("step:", i+1)
        print("action:", action)
        print("obs:", obs)
        print("reward:", reward)
        print("done:", bdone)

if __name__ == "__main__":
    main()