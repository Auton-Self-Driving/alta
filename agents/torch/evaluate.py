import os

import numpy as np
import hydra

from environment.carla_9_4.env import CarlaEnv


EXPERIMENT_DIR = '/home/scratch/brianyan/outputs/sac/2021-03-25_14-39-28'
CHECKPOINT = 'epoch=69-step=3499.ckpt'



@hydra.main(config_name='{}/.hydra/config.yaml'.format(EXPERIMENT_DIR))
def main(cfg):
    agent = hydra.utils.instantiate(cfg.algo.agent)
    agent.load_from_checkpoint('{}/checkpoints/{}'.format(EXPERIMENT_DIR, CHECKPOINT), **cfg.algo.agent)
    agent = agent.cuda().eval()

    reward_list = []
    status_list = []

    env = CarlaEnv(log_dir=os.getcwd(), server_port=41456, **cfg.environment)
    for index in range(1):
        obs = env.reset(unseen=True, index=index)
        total_reward = 0.

        for _ in range(10000):
            action = np.array([0,-.5]) # agent.predict(obs)[0]
            obs, reward, done, info = env.step(action)

            total_reward += reward

            # frame = env.render()
            # cv2.imshow('frame', frame)
            # cv2.waitKey(.01)

            print('=======')
            print(info)

            if done:
                break

        reward_list.append(total_reward)
        status_list.append(info['termination_state'])
        print(status_list[-1])

        # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()