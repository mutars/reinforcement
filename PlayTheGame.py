from collections import deque

from atari.pytorch.NNet import NNetWrapper
from atari.AtariGame import AtariGame
from mcts import MCTSNode
from GameSession import GameSession


game_i = AtariGame()
view = game_i.getInitBoard()
nnet_checkpoint = NNetWrapper(game_i)
mtsc_root = MCTSNode(view, action_size=game_i.getActionSize())

nnet_checkpoint.load_checkpoint()

gameSession = GameSession(nnet_checkpoint, game_i, mtsc_root)
max_value = -1
target_endpoint_leafs = []
for x in range(500):
    gameSession.execute_episode()
    print(gameSession.max_score)
    if gameSession.max_score > max_value:
        max_value = gameSession.max_score
        target_endpoint_leafs = [gameSession.game_end_leaf]
    elif gameSession.max_score == max_value and len(target_endpoint_leafs) < 50 and gameSession.game_end_leaf not in target_endpoint_leafs:
        target_endpoint_leafs.append(gameSession.game_end_leaf)


train_examples  =  []
for enpoint_node in target_endpoint_leafs:
    train_examples += gameSession.makeExamples(enpoint_node, max_value)

#gameSession.play_game(render=True)

nnet_checkpoint.train(train_examples)

nnet_checkpoint.save_checkpoint()


