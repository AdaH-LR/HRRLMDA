
import pickle
import numpy as np
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from datapro import Simdata_pro, loading_data
from train import train_test,Loader
ENV_MOVE = False
class Cube:
    def __init__(self, size, x=None, y=None):
        self.size = size
        if x is not None and y is not None:
            self.x = x
            self.y = y
        else:
            self.x = np.random.randint(0, self.size)
            self.y = np.random.randint(0, self.size)

    def create_enemy_at_position(self, x, y):
        enemy = Cube(self.size, x, y)
        return enemy

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __str__(self):
        return f'{self.x},{self.y}'

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
class envCube:
    NUM_PLAYERS = 1
    OBSERVATION_SPACE_VALUES = 128*128
    ACTION_SPACE_VALUES = 6
    FOOD_REWARD = 100
    ENEMY_PENALITY = -10
    MOVE_PENALITY = -4
    CLOSER_REWARD = 5
    FARER_PENALITY = -5
    STAY_REWARD = 0
    def reset(self, c, item1, simData, train_data, param, trainLoader,test_edges,test_labels, model):
        pre_score, result, loss, c = train_test(c, train_data, item1, simData, param, trainLoader,test_edges,test_labels, model)
        return pre_score, result, loss, c
    def step(self, a, i, item1, simData, train_data, param,trainLoader,test_edges,test_labels, model, state, action, loss):
        env = envCube()
        if a <= 1:
            self.old_loss = loss
        else:
            self.old_loss = self.old_loss
        pre_score, state, loss, c = env.reset(i, item1, simData, train_data, param, trainLoader, test_edges, test_labels, model)
        if self.old_loss > loss:
            reward = self.CLOSER_REWARD
        elif self.old_loss < loss:
            reward = self.FARER_PENALITY
        else:
            reward = self.STAY_REWARD
        self.old_loss = loss
        new_state = state
        return new_state, reward, loss, c
    def step1(self, action, param):
        if action == 0:
            param.nodeNum = 8
        elif action == 1:
            param.nodeNum = 16
        elif action == 2:
            param.nodeNum = 32
        elif action == 3:
            param.nodeNum = 64
        elif action == 4:
            param.nodeNum = 128
        elif action == 5:
            param.nodeNum = 256
        return param

