import gym_super_mario_bros 
from nes_py.wrappers import JoypadSpace
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from models import InverseDyanmicsModel
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision import transforms as T
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

def train(state, next_state, labels, plot=True):
    
    data = np.concatenate((state, next_state), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    trainData = DataLoader(X_train, batch_size=1, shuffle=True)
    trainLabels = DataLoader(y_train, batch_size=1, shuffle=True)
    testData = DataLoader(X_test, batch_size=1, shuffle=True)
    testLabels = DataLoader(y_test, batch_size=1, shuffle=True)

    model = InverseDyanmicsModel().to('cuda')

    NUM_EPOCHS = 1000
    LR = 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    acc_list = []
    loss_list = []

    for epoch in (pbar := tqdm(range(NUM_EPOCHS))):
        model.train()

        train_correct = 0
        train_loss = 0.0
        for i, (data, labels) in enumerate(zip(trainData, trainLabels)):

            data = torch.tensor(data, dtype=torch.float32).to("cuda")
            labels = torch.tensor(labels, dtype=torch.long).to("cuda")

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_correct += (output.argmax(dim=1) == labels).sum().item()
            train_loss += loss.item()

        # train_acc /= len(trainData)
        train_acc = train_correct / len(trainData)
        acc_list.append(train_acc)
        train_loss /= len(trainData)
        loss_list.append(train_loss)
        
        pbar.set_description(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        ax[0].plot(acc_list)
        ax[0].set_title("Training Accuracy")
        ax[1].plot(loss_list)
        ax[1].set_title("Training Loss")
        fig.savefig("training.png")
        plt.close(fig)

    return acc_list, loss_list

if __name__ == "__main__":
    embeddings = np.load("embeddings.npy", allow_pickle=True)
    # embeddings is a list of dicts with keys 'embedding' and 'action', converted to numpy arrays of 'data' and 'labels' respectively
    labels = np.array([i['action'] for i in embeddings])
    state = np.array([i['state'] for i in embeddings])
    next_state = np.array([i['next_state'] for i in embeddings])

    acc, loss = train(state, next_state, labels)