import numpy as np
import random
import collections

TEMPERATURE_CUTOFF = int(20000)


class GameSession:
    # per one thread synchro
    def __init__(self, nnet, game, mcts_root):
        self.nnet = nnet
        self.game = game
        self.mcts_root = mcts_root
        self.game_end_leaf = None
        self.max_score = 0

    def init_game(self):
        self.game.getInitBoard()

    def execute_episode(self):
        self.init_game()
        cur_node = self.mcts_root
        while True:
            cur_node = cur_node.select_leaf(lambda action: self.game.getNextState(action))
            if self.game.isDone():
                self.max_score = self.game.getTotalScore()
                cur_node.backup_total_value(np.tanh(np.log(self.max_score)), up_to=self.mcts_root)
                return cur_node
            elif not cur_node.is_expanded:
                move_prob, value = self.nnet.predict(self.get_cum_view(cur_node, cur_node.view.shape))
                cur_node.incorporate_results(move_prob, value, up_to=self.mcts_root)

    def play_game(self, render=False):
        self.init_game()
        node = self.mcts_root
        step = 0
        while True:
            step += 1
            action = self.pick_move(node, step)
            node = self.play_move(node, step, action)
            if render:
                self.game.env.render()
            if self.game.isDone():
                return node, self.game.getTotalScore()

    def play_move(self, node, step, action):
        view, _, _ = self.game.getNextState(action)
        next_node = node.maybe_add_child(action, view)
        if not next_node.is_expanded:
            move_prob, t_value = self.nnet.predict(self.get_cum_view(next_node, next_node.view.shape))
            next_node.incorporate_results(move_prob, t_value, up_to=self.mcts_root)
        #pi = next_node.children_as_pi(step < TEMPERATURE_CUTOFF)
        return next_node

    def pick_move(self, node, step):
        '''Picks a move to play, based on MCTS readout statistics.
        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max.'''
        if step > TEMPERATURE_CUTOFF:
            action = np.argmax(node.child_N)
        else:
            cdf = node.child_N.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            action = cdf.searchsorted(selection)
        return action

    def get_cum_view(self, node, shape):
        result = []
        for x in range(8):
            if node.parent is not None:
                result += [node.view]
                node = node.parent
            else:
                result += [np.zeros(shape, dtype=np.uint8)]
        return result

    '''def get_cum_view_from_queue(self, queue):
        result = queue.popleft()
        shape = result.shape
        for x in range(7):
            if len(queue) > 0:
                result = np.dstack((result, queue.popleft()))
            else:
                result = np.dstack((result, np.zeros(shape, dtype=np.uint8)))
        return result'''
