from Game import Game
import numpy as np
import gym


class AtariGame(Game):

    def __init__(self, env_name='Breakout-v0', seed = 0):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.env.env.frameskip = 3
        self.env.env.ale.setInt(b'random_seed', 1)
        self.env.env.ale.loadROM(self.env.env.game_path)

        self.done = False
        self.observation_shape = self.env.observation_space.shape
        self.total_reward = 0

    def getInitBoard(self):
        self.total_reward = 0
        self.done = False
        self.env.reset()
        observation, reward, done, info = self.env.step(1)
        return observation

    def getBoardSize(self):
        # (a,b) tuple
        return self.observation_shape

    def getActionSize(self):
        # return number of actions
        return 3

    def getNextState(self, action):
        move = self.getValidMoves()[action]
        #print(move)
        observation, reward, done, info = self.env.step(move)
        self.env.step(1)
        self.total_reward += reward
        self.done = True if info['ale.lives'] < 5 else False
        return observation, reward, done

    def getValidMoves(self):
        return [0,2,3]

    def isDone(self):
        return self.done

    def getSymmetries(self, board, pi):
        #no symmetreis here
        return (board,pi)

    def getTotalScore(self):
        return self.total_reward

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, observation):
        return observation.tostring()
