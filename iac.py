import numpy as np
from expert import Expert
import torch
from sklearn.preprocessing import OneHotEncoder
from time import time

class IAC:
    def __init__(self, env, sensory_dim=84*84):
        self.env = env
        self.state = self.env.reset()
        self.action_space = self.env.action_space
        self.sensory_dim = sensory_dim

        self.random_steps = 0
        self.repeat_action = 4

        self.num_steps = 0

        self.n_regions = 1
        self.centroids = np.zeros((1, self.sensory_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experts = [Expert().to(self.device)]
        self.state_motor = [[]]
        self.next_state = [[]]
        self.error = [[]]
    
    # def transform_obs(self, obs):
    #     obs = ResizeObservation(obs, (84, 84)).observation
    #     obs = GrayScaleObservation(obs).observation
        # return obs
    
    def reset(self):
        self.state = self.env.reset()
        return self.state
    
    def render(self):
        self.env.render()
    
    def select_region(self):
        dists = np.linalg.norm(self.centroids - self.state.flatten(), axis=1)
        return np.argmin(dists)

    def select_action(self, region):

        # choose a random action for the first self.num_steps
        if self.num_steps < self.random_steps:
            return self.env.action_space.sample()

        # list of learning progress values for each action
        lp_list = np.zeros(self.action_space.n)

        for i, action in enumerate(list(range(self.action_space.n))):
            # TODO: implement the observation space feature encoding here
            SM_t = np.concatenate((self.state.flatten(), np.array([action])))

            # determine expected learning progress
            lp_list[i] = self.calculate_lp(region, self.state, action)
        
        action = np.argmax(lp_list)
        return action
    
    def calculate_lp(self, region, state, action):
        return None
    
    def predict_state(self, region, action):
        self.experts[region].eval()
        action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=self.action_space.n).float()
        X = torch.cat((torch.tensor(self.state.flatten()), action)).to(self.device)
        return self.experts[region](X).detach().cpu().numpy()
    
    def update_expert(self, region, state, action, next_state):
        self.experts[region].train()
        action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=self.action_space.n).float()
        X = torch.cat((torch.tensor(state.flatten()), action)).to(self.device)
        y = torch.tensor(next_state.flatten()).float().to(self.device)
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.experts[region].parameters(), lr=0.001)
        optimizer.zero_grad()
        output = self.experts[region](X)
        loss = loss(output, y)
        loss.backward()
        optimizer.step()
    
    def step(self):
        # selection region and action
        region = self.select_region()
        action = self.select_action(region)
        self.state_motor[region].append((self.state, action))

        # make prediction
        predicted_state = self.predict_state(region, action)

        # step environment
        next_state, reward, done, info = self.env.step(action)
        self.next_state[region].append(next_state)

        self.num_steps += 1

        # update error of the region
        self.error[region].append(np.linalg.norm(predicted_state - next_state.flatten()))
        self.update_expert(region, self.state, action, next_state)
        

            

if __name__ == "__main__":
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
    from gym.wrappers.resize_observation import ResizeObservation
    from gym.wrappers.gray_scale_observation import GrayScaleObservation


    env = gym_super_mario_bros.make('SuperMarioBros-v0', new_step_api=False)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)

    iac = IAC(env)

    for i in range(100):
        start = time()
        iac.step()
        print(time() - start)