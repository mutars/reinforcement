from atari.pytorch.NNet import NNetWrapper
from atari.AtariGame import AtariGame
from MCTS_GGL import MCTSNode
from GameSession import GameSession


game_i = AtariGame()
view = game_i.getInitBoard()
nnet_checkpoint = NNetWrapper(game_i)
mtsc_root = MCTSNode(view, action_size=game_i.getActionSize())

#nnet_checkpoint.load_checkpoint()

gameSession = GameSession(nnet_checkpoint, game_i, mtsc_root)
for x in range(1000):
    gameSession.execute_episode()
    print(gameSession.max_score)



