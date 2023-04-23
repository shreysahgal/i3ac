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
import cv2

def generate_data(env, num = 1000, repeated_actions = 6):
    data = []
    labels = []
    prev_state = env.reset()
    for i in tqdm(range(num)):
        action = env.action_space.sample()
        for step in range(repeated_actions):
            new_state,_,_,_ = env.step(action)
        labels.append(action)
        if np.array_equal(prev_state, new_state):
            print("same")
        data.append(np.stack((prev_state, new_state)))
        prev_state = new_state

    env.reset()
    return data, labels

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

data, labels = generate_data(env)

data = np.array(data)
labels = np.array(labels)

for i in range(0,1000,10):
    print(labels[i])
    cv2.imshow('s(t)', data[i][0])
    cv2.waitKey(0)
    cv2.imshow('s(t+1)', data[i][1])
    cv2.waitKey(0)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

trainData = DataLoader(X_train, batch_size=1, shuffle=True)
trainLabels = DataLoader(y_train, batch_size=1, shuffle=True)
testData = DataLoader(X_test, batch_size=1, shuffle=True)
testLabels = DataLoader(y_test, batch_size=1, shuffle=True)

model = InverseDyanmicsModel().to('cuda')

NUM_EPOCHS = 100
LR = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
        
        train_correct += (output.argmax(1) == labels).sum().item()
        train_loss += loss.item()
        
        # acc.append(train_acc)

    # train_acc /= len(trainData)
    train_acc = train_correct / len(trainData)
    acc_list.append(train_acc)
    train_loss /= len(trainData)
    loss_list.append(train_loss)

    print(train_acc)
    # pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")
        

fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].plot(acc)
ax[0].set_title("Training Accuracy")
ax[1].plot(loss)
ax[1].set_title("Training Loss")
fig.savefig("training.png")