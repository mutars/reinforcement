from Game import Game
import numpy as np
import gym


class AtariGame(Game):

    def __init__(self, env_name='Breakout-v0', state = None):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.state = state
        if state is None:
            self.state = self.env.unwrapped.clone_full_state()

        self.env.unwrapped.frameskip = 4
        self.done = False
        self.observation_shape = (self.env.observation_space.shape[0],self.env.observation_space.shape[1],1)
        self.total_reward = 0

    def getInitBoard(self):
        self.total_reward = 0
        self.done = False
        self.env.reset()
        self.env.unwrapped.restore_full_state(self.state)
        self.env.step(1)
        return self.env.unwrapped.ale.getScreenGrayscale()

    def getBoardSize(self):
        # (a,b) tuple
        return self.observation_shape

    def getActionSize(self):
        # return number of actions
        return 3

    def getNextState(self, action):
        #print(action, end='', flush=True)
        #print(' ', end='', flush=True)
        move = self.getValidMoves()[action]
        observation, reward, _, info = self.env.step(move)
        self.env.step(1)
        self.total_reward += reward
        self.done = True if info['ale.lives'] < 5 else False
        return self.env.unwrapped.ale.getScreenGrayscale(), reward, self.done

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
