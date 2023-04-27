import numpy as np
from models import Expert
import torch
from sklearn.preprocessing import OneHotEncoder
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gc
from tabulate import tabulate

# TODO: add region splitting DONE
# TODO: repeated actions DONE
# TODO: calculate LP


class Region:

    def __init__(self, max_samples=100, centroids=None, tau=15, theta=25, random_steps=100, n_parent_samples=0, image_dims=(84,84), action_space_dim=12):

        if centroids is None:
            self.centroids = np.zeros((1, 84*84 + action_space_dim))
        else:
            self.centroids = centroids
        self.action_space_dim = action_space_dim

        self.max_samples = max_samples
        self.n_parent_samples = n_parent_samples

        self.learning_progress = np.inf
        self.tau = tau
        self.theta = theta
        self.random_steps = random_steps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expert = Expert(obs_dim=image_dims[0]*image_dims[1], action_space_dim=action_space_dim)
        self.optimizer = torch.optim.Adam(self.expert.parameters(), lr=0.001)

        self.state_motor = np.zeros((self.max_samples + self.n_parent_samples, 84*84 + action_space_dim))
        self.next_state = np.zeros((self.max_samples + self.n_parent_samples, 84*84))
        self.error = np.zeros(self.max_samples + self.n_parent_samples)
        self.num_state_motor = self.n_parent_samples
        self.num_next_state = self.n_parent_samples
        self.num_error = self.n_parent_samples
    
    def train_on_parent_samples(self, state_motor, next_state, num_epochs=100):

        assert self.n_parent_samples != 0
        
        self.expert.train()
        self.expert.to(self.device)
        X = torch.tensor(state_motor).float().to(self.device)
        y = torch.tensor(next_state).float().to(self.device)
        
        data = torch.utils.data.TensorDataset(X, y)
        data_loader = torch.utils.data.DataLoader(data, batch_size=int(self.n_parent_samples), shuffle=True)

        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()

        for epoch in tqdm(range(num_epochs)):
            for batch_idx, (data, target) in enumerate(data_loader):
                output = self.expert(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                error = np.linalg.norm(output.detach().cpu().numpy() - target.cpu().numpy(), axis=1)
                error_from_parent = error

        self.expert.to("cpu")

        if np.all(error == 0):
            breakpoint()
        self.error[:self.n_parent_samples] = error
        self.update_learning_progress()

    
    def add_state_motor(self, state, action):
        action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=self.action_space_dim).float()
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
    
    def update_learning_progress(self):
        # if self.num_error <= self.random_steps:
        #     self.learning_progress = 0
        
        if self.num_error <= self.tau + self.theta + 1:
            # only true for the first region
            print("Not enough samples and is the first region")
            assert self.tau + self.theta + 1 <= self.random_steps
            self.learning_progress = np.inf
        
        else:
            if self.num_error - self.n_parent_samples >= self.tau + self.theta:
                print("Enough samples... calculating LP with all samples")
            else:
                print("Not eough samples... calculating LP with parents and region samples")
            e1 = np.sum(self.error[self.num_error-self.theta-1:self.num_error-1])
            e2 = np.sum(self.error[self.num_error-self.tau-self.theta-1:self.num_error-1])
            self.learning_progress = (e2 - e1) / self.theta
        
        if self.learning_progress == 0:
            breakpoint()
        
    def get_last_error(self):
        return self.error[self.num_error-1]
    
    def get_num_samples(self):
        assert self.num_state_motor == self.num_next_state == self.num_error
        return self.num_state_motor

    def split(self):
        self.expert.to("cpu")
        del self.expert

        state_motor = self.state_motor[self.n_parent_samples:]
        next_state = self.next_state[self.n_parent_samples:]

        kmeans = KMeans(n_clusters=2, random_state=0).fit(state_motor)
        centroids = kmeans.cluster_centers_
        clusters = kmeans.labels_

        region1 = Region(self.max_samples, centroids[0], n_parent_samples=np.sum(clusters==0), action_space_dim=self.action_space_dim)
        region2 = Region(self.max_samples, centroids[1], n_parent_samples=np.sum(clusters==1), action_space_dim=self.action_space_dim)

        if np.all(next_state[clusters == 0] == 0) or np.all(next_state[clusters == 1] == 0):
            breakpoint()

        print(f"Split 0: {np.sum(clusters == 0)}, Split 1: {np.sum(clusters == 1)}")
        region1.train_on_parent_samples(state_motor[clusters == 0], next_state[clusters == 0])
        region2.train_on_parent_samples(state_motor[clusters == 1], next_state[clusters == 1])

        return region1, region2
    
    def should_split(self):
        assert self.num_state_motor == self.num_next_state == self.num_error
        if self.num_state_motor >= self.max_samples + self.n_parent_samples:
            return True
        else:
            return False
        

class IAC:
    def __init__(self, env, sensory_dim=84*84, split_threshold=100, epsilon=0.2, render=False):
        self.env = env
        self.render = render
        self.state = self.env.reset()
        self.action_space = self.env.action_space
        self.sensory_dim = sensory_dim

        self.random_steps = 50
        self.repeat_action = 6
        self.epsilon = epsilon

        self.num_steps = 0

        self.n_regions = 1
        self.regions = [Region(action_space_dim=self.action_space.n)]
        self.prev_region = None

        self.split_threshold = split_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optim = torch.optim.Adam(self.regions[0].expert.parameters(), lr=0.001)
    
    def reset(self):
        self.state = self.env.reset()
        return self.state

    def select_region_action(self):

        # choose a random action for the first self.num_steps
        # with probability epsilon, choose a random action

        if self.num_steps < self.random_steps or np.random.rand() < self.epsilon:
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
                # break argmin ties randomly
                region = np.random.choice(np.where(dists == np.min(dists))[0])
            return region, action
        
        # choose action based on learning progress
        else:
            # list of learning progress values for each action
            lp_list = np.zeros(self.action_space.n)
            regions = np.zeros(self.action_space.n)

            for i, action in enumerate(list(range(self.action_space.n))):
                # create SM(t) vector
                action_vec = np.zeros(self.action_space.n)
                action_vec[action] = 1
                SM_t = np.concatenate((self.state.flatten(), action_vec))
                # for each region in self.regions calculate the distance to the state-action vector, find min dists
                dists = np.zeros(self.n_regions)
                for j in range(self.n_regions):
                    dists[j] = np.linalg.norm(self.regions[j].centroids - SM_t)
                region = np.argmin(dists)
                regions[action] = region
                # determine expected learning progress
                lp_list[action] = self.regions[region].learning_progress

                if self.n_regions > 1:
                    print(f"{action}: {region}, {self.regions[region].learning_progress}")
            
            # action = int(np.argmax(lp_list))
            # break argmax ties randomly
            action = np.random.choice(np.where(lp_list == np.max(lp_list))[0])
            region = int(regions[action])

            # if self.n_regions > 1:
            #     breakpoint()

            return region, action
    
    def calculate_lp(self, region, state, action):
        return None

    def step(self):
        # selection region and action
        region_idx, action = self.select_region_action()
        
        if region_idx != self.prev_region:
            # unload prev region expert off of gpu
            if self.prev_region is not None:
                self.regions[self.prev_region].expert.to('cpu')
                print(f"unloading region {self.prev_region} expert off of gpu")
            # load region expert onto gpu
            self.regions[region_idx].expert.to('cuda')
            print(f"loading region {region_idx} expert onto gpu")
        self.prev_region = region_idx

        # update region state motor
        self.regions[region_idx].add_state_motor(self.state, action)
        # make prediction
        predicted_state = self.regions[region_idx].predict_state()
        # step environment
        for _ in range(self.repeat_action):
            next_state, reward, done, info = self.env.step(action)
            if self.render:
                self.env.render()
            if done:
                self.env.reset()
        # update region next state
        self.regions[region_idx].add_next_state(next_state)
        # update error of the region
        error = np.linalg.norm(predicted_state - next_state.flatten())
        self.regions[region_idx].update_error(error)
        # print(len(error_list))gion_idx].update_error(error)
        # update region expert
        self.regions[region_idx].update_expert()
        self.regions[region_idx].update_learning_progress()

        # print times
        # s = ""
        # for i in range(len(times)-1):
        #     s += str(times[i+1] - times[i]) + ", "
        # print(s)

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
            self.prev_region = None

            # breakpoint()

        gc.collect()
        torch.cuda.empty_cache()
        
        if self.num_steps % 20 == 0:
            data = [
                ["Region:"] + [i for i in range(self.n_regions)],
                ["Num Steps:"] + [f"{region.num_error}/{region.max_samples + region.n_parent_samples}" for region in self.regions],
                ["LP:"] + [region.learning_progress for region in self.regions],
                ["Error:"] + [region.get_last_error() for region in self.regions]
            ]
            print(tabulate(data, headers="firstrow", tablefmt="fancy_grid"))
            # s1 = "region:\t"
            # s2 = "num_steps:\t"
            # s3 = "LP:\t"
            # s4 = "error:\t"
            # for i, region in enumerate(self.regions):
            #     s1 += f"{i}\t"
            #     s2 += f"{region.num_error}\t"
            #     s3 += f"{region.learning_progress}\t"
            #     s4 += f"{region.get_last_error()}\t"
            # print(s1)
            # print(s2)
            # print(s3)
            # print(s4)

        self.state = next_state
        self.num_steps += 1
        
        return error


if __name__ == "__main__":
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
    from gym.wrappers.resize_observation import ResizeObservation
    from gym.wrappers.gray_scale_observation import GrayScaleObservation


    env = gym_super_mario_bros.make('SuperMarioBros-v0', new_step_api=False)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)

    iac = IAC(env, render=False)

    error_list = []

    for i in range(1000):
        start = time()
        # iac.render()
        error = iac.step()
        error_list.append(error)
        # print(error)
        # print(len(error_list))
        print(f"time: {time() - start}")
        print()
    
    plt.plot(error_list[::10])
    plt.yscale("log")
    plt.savefig("asdf.png")