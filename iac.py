import numpy as np
from expert import Expert
import torch
from sklearn.preprocessing import OneHotEncoder
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gc

# TODO: add region splitting
# TODO: repeated actions DONE
# TODO: calculate LP

class Region:

    def __init__(self, max_samples=100, centroids=np.zeros((1, 84*84 + 12))):

        self.max_samples = max_samples
        self.centroids = centroids
        self.num_state_motor = 0
        self.num_next_state = 0
        self.num_error = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expert = Expert().to(self.device)
        self.optimizer = torch.optim.Adam(self.expert.parameters(), lr=0.001)

        self.state_motor = np.zeros((self.max_samples, 84*84 + 12))
        self.next_state = np.zeros((self.max_samples, 84*84))
        self.error = np.zeros(self.max_samples)
    
    def inherit_from_parent(self, state_motor, next_state, num_samples):
        self.state_motor[:num_samples] = state_motor
        self.next_state[:num_samples] = next_state
        print(num_samples)
        self.num_state_motor = num_samples
        self.num_next_state = num_samples
        self.num_error = num_samples

        # TODO: train child expert on given samples
    
    def train_on_parent_samples(self, state_motor, next_state, num_samples, num_epochs=100):
        self.expert.train()
        X = torch.tensor(state_motor).float().to(self.device)
        y = torch.tensor(next_state).float().to(self.device)
        
        data = torch.utils.data.TensorDataset(X, y)
        data_loader = torch.utils.data.DataLoader(data, batch_size=int(num_samples), shuffle=True)

        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()

        for epoch in tqdm(range(num_epochs)):
            for batch_idx, (data, target) in enumerate(data_loader):
                output = self.expert(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    
    def add_state_motor(self, state, action):
        action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=12).float()
        state_motor = np.concatenate((state.flatten(), action))
        self.state_motor[self.num_state_motor] = state_motor
        self.num_state_motor += 1
    
    def add_next_state(self, next_state):
        self.next_state[self.num_next_state] = next_state.flatten()
        self.num_next_state += 1
    
    def predict_state(self):
        self.expert.eval()
        X = torch.tensor(self.state_motor[self.num_state_motor-1]).float().to(self.device)
        output = self.expert(X).detach().cpu().numpy()
        X.to("cpu")
        del X
        return output

    def update_expert(self):
        self.expert.train()
        X = torch.tensor(self.state_motor[self.num_state_motor-1]).float().to(self.device)
        y = torch.tensor(self.next_state[self.num_next_state-1]).float().to(self.device)

        loss = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        output = self.expert(X)
        loss = loss(output, y)
        loss.backward()
        self.optimizer.step()
        X.to("cpu")
        y.to("cpu")
        del X, y
    
    def update_error(self, error):
        self.error[self.num_error] = error
        self.num_error += 1
        
    def get_last_error(self):
        return self.error[self.num_error-1]
    
    def get_num_samples(self):
        assert self.num_state_motor == self.num_next_state == self.num_error
        return self.num_state_motor

    def split(self):
        del self.expert
        kmeans = KMeans(n_clusters=2, random_state=0).fit(self.state_motor)
        centroids = kmeans.cluster_centers_
        clusters = kmeans.labels_

        region1 = Region(self.max_samples, centroids[0])
        region2 = Region(self.max_samples, centroids[1])

        # region1.inherit_from_parent(self.state_motor[clusters == 0], self.next_state[clusters == 0], np.sum(clusters == 0))
        # region2.inherit_from_parent(self.state_motor[clusters == 1], self.next_state[clusters == 1], np.sum(clusters == 1))

        region1.train_on_parent_samples(self.state_motor[clusters == 0], self.next_state[clusters == 0], np.sum(clusters==0))
        region2.train_on_parent_samples(self.state_motor[clusters == 1], self.next_state[clusters == 1], np.sum(clusters==1))

        return region1, region2
    
    def should_split(self):
        if self.num_state_motor >= self.max_samples:
            return True
        else:
            return False
    
    def expert_to_gpu(self):
        self.expert.to(self.device)
    
    def expert_to_cpu(self):
        self.expert.to("cpu")

        

class IAC:
    def __init__(self, env, sensory_dim=84*84, split_threshold=100):
        self.env = env
        self.state = self.env.reset()
        self.action_space = self.env.action_space
        self.sensory_dim = sensory_dim

        self.random_steps = 1000
        self.repeat_action = 2

        self.num_steps = 0

        self.n_regions = 1
        self.regions = [Region()]

        self.split_threshold = split_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optim = torch.optim.Adam(self.regions[0].expert.parameters(), lr=0.001)
    
    def reset(self):
        self.state = self.env.reset()
        return self.state
    
    def render(self):
        self.env.render()

    def select_region_action(self):

        # choose a random action for the first self.num_steps
        if self.num_steps < self.random_steps:
            action = self.env.action_space.sample()
            if self.n_regions < 2:
                region = 0
            else:
                action_vec = np.zeros(self.action_space.n)
                action_vec[action] = 1
                SM_t = np.concatenate((self.state.flatten(), action_vec))
                # for each region in self.regions calculate the distance to the state-action vector, find min dists
                dists = np.zeros(self.n_regions)
                for i in range(self.n_regions):
                    dists[i] = np.linalg.norm(self.regions[i].centroids - SM_t)
                region = np.argmin(dists)
            return region, action

        # TODO: update this with new regions
        """ 
        # list of learning progress values for each action
        lp_list = np.zeros(self.action_space.n)

        for i, action in enumerate(list(range(self.action_space.n))):
            # TODO: implement the observation space feature encoding here
            action_vec = np.zeros(self.action_space.n)
            action_vec[action] = 1
            SM_t = np.concatenate((self.state.flatten(), action_vec))

            dists = np.linalg.norm(self.centroids - SM_t, axis=1)
            region = np.argmin(dists)

            # determine expected learning progress
            lp_list[i] = self.calculate_lp(region, self.state, action)
        
        action = np.argmax(lp_list)
        return region, action
        """
    
    def calculate_lp(self, region, state, action):
        return None

    def step(self):
        # selection region and action
        region_idx, action = self.select_region_action()

        # load region expert onto gpu
        # self.regions[region_idx].expert.to('cuda')

        # update region state motor
        self.regions[region_idx].add_state_motor(self.state, action)

        # make prediction
        predicted_state = self.regions[region_idx].predict_state()

        # step environment
        for _ in range(self.repeat_action):
            next_state, reward, done, info = self.env.step(action)

        # update region next state
        self.regions[region_idx].add_next_state(next_state)

        # update error of the region
        error = np.linalg.norm(predicted_state - next_state.flatten())
        self.regions[region_idx].update_error(error)

        # update region expert
        self.regions[region_idx].update_expert()

        # load region expert off of gpu
        # self.regions[region_idx].expert.to('cpu')

        print("Step: {}, Region: {}, Action: {}, Error: {}".format(self.num_steps, region_idx, action, self.regions[region_idx].get_last_error()))

        # print cached, allocated, and total memory in GB
        # print("Cached Memory: {}, Allocated Memory: {}, Total Memory: {}".format(torch.cuda.memory_cached()/1e9, torch.cuda.memory_allocated()/1e9, torch.cuda.get_device_properties(0).total_memory/1e9))

        # split region if number of samples is greater than threshold
        if self.regions[region_idx].should_split():
            print(f"splitting region {region_idx}")
            region1, region2 = self.regions[region_idx].split()
            self.regions[region_idx] = region1
            self.regions.append(region2)
            self.n_regions += 1

        gc.collect()
        torch.cuda.empty_cache()
        
        self.num_steps += 1
        
        return error



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

    error_list = []

    for i in range(1000):
        start = time()
        error = iac.step()
        error_list.append(error)
        # print(error)
        # print(len(error_list))
        print(f"time: {time() - start}")
        print()
    
    plt.plot(error_list[::10])
    plt.yscale("log")
    plt.savefig("asdf.png")