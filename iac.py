import numpy as np

class IAC:
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.random_steps = 100
        self.repeat_action = 4

        self.num_steps
        self.n_regions = 1
        self.state_motor = []
        self.next_state = []
        self.error = []
        self.mode_probs = {1: 0.3, 2: 0.6, 3: 0.1}  # TODO: maybe make these change over time?
    
    # def transform_obs(self, obs):
    #     obs = ResizeObservation(obs, (84, 84)).observation
    #     obs = GrayScaleObservation(obs).observation
        # return obs
    
    def reset(self):
        self.state = self.env.reset()
        return self.state
    
    def render(self):
        self.env.render()

    def select_action(self):

        # choose a random action for the first self.num_steps
        if self.num_steps < self.random_steps:
            return self.env.action_space.sample()

        # choose mode
        mode = np.random.choice(list(self.mode_probs.keys()), p=list(self.mode_probs.values()))

        for action in list(range(self.action_space.n)):
            # TODO: implement the observation space feature encoding here
            SM_t = np.concatenate((self.state.flatten(), np.array([action])))

            # choose a region
            if self.n_regions == 1:
                region = 0
            
            else:
                pass
            
            # choose an action
            action = None

            # mode 1: babble
            if mode == 1:
                action = np.random.choice(list(range(self.action_space.n)))

            # mode 2: LP maximization
            if mode == 2:
                pass

            # mode 3: error maximization
            if mode == 3:
                pass

            

if __name__ == "__main__":
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
    from gym.wrappers.resize_observation import ResizeObservation
    from gym.wrappers.gray_scale_observation import GrayScaleObservation


    env = gym_super_mario_bros.make('SuperMarioBros-v0', new_step_api=True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)

    iac = IAC(env)
    iac.select_action()