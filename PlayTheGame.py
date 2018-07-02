from collections import deque

import numpy as np

from atari.pytorch.NNet import NNetWrapper
from atari.AtariGame import AtariGame
from mcts import MCTSNode
from GameSession import GameSession

def prepareTestSet(nodes):
    s = set()
    r = []
    for (node, value) in nodes:
        n = node
        while n.parent.parent is not None:
            if n not in s:
                r += [(n, value)]
                s.add(n)
                n = n.parent
            else:
                break
    return r


game_i = AtariGame()
view = game_i.getInitBoard()
nnet_checkpoint = NNetWrapper(game_i)
mtsc_root = MCTSNode(view, action_size=game_i.getActionSize())

nnet_checkpoint.load_checkpoint()

gameSession = GameSession(nnet_checkpoint, game_i, mtsc_root)
gameSession.execute_episode()
threshold = 0
print("game score={0}".format(threshold))
for x in range(2000):
    node = gameSession.execute_episode()
    #print('')
    if gameSession.max_score > threshold:
        print("max score={0}".format(gameSession.max_score))
        threshold = gameSession.max_score


nodes = []
for y in range(500):
    node, val = gameSession.play_game()
    if val >= threshold:
        print("mtsc score={0}".format(val))
        nodes += [(node, val)]
resultSet = prepareTestSet(nodes)
result = [(gameSession.get_cum_view(node.parent, node.view.shape), node.action , np.tanh(np.log(val)) ) for (node, val) in resultSet]

if len(result) > 0:
    nnet_checkpoint.train(result)
    nnet_checkpoint.save_checkpoint()

#gameSession.play_game(render=True)



