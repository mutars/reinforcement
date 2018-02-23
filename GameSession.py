import numpy as np


class GameSession:


    #per one thread synchro
    def __init__(self, nnet, game, mctc_root):
        self.nnet = nnet
        self.game = game
        self.mctc_root = mctc_root
        self.game_end_leaf = None
        self.max_score = 0

    def init_game(self):
        for x in range(10):
            view = self.game.getInitBoard()
            if np.array_equal(self.mctc_root.view,view):
                break

    def execute_episode(self):
        self.init_game()
        cur_node = self.mctc_root
        while True:
            cur_node, reward = cur_node.select_leaf(self.game)
            # if game is over, override the value estimate with the true score
            if self.game.isDone():
                value = np.tanh(self.game.getTotalScore())
                self.max_score = self.game.getTotalScore()
                cur_node.backup_total_value(value, up_to=self.mctc_root)
                break
            elif not cur_node.is_expanded:
                #leaf.add_virtual_loss(up_to=self.root)
                move_prob, t_value = self.nnet.predict(self.get_cum_view(cur_node))
                cur_node.incorporate_results(move_prob, t_value, up_to=self.mctc_root)

    def get_cum_view(self, node):
        result = node.view
        shape = result.shape
        leaf = node
        for x in range(7):
            if leaf.parent.parent is not None:
                result = np.dstack((result, leaf.parent.view))
                leaf = leaf.parent
            else:
                result = np.dstack((result, np.zeros(shape)))
        return result


