import numpy as np
import random
import collections

TEMPERATURE_CUTOFF = int(0)

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
                value = self.game.getTotalScore() / (1 + self.game.getTotalScore())
                self.max_score = self.game.getTotalScore()
                cur_node.backup_total_value(value, up_to=self.mctc_root)
                break
            elif not cur_node.is_expanded:
                #leaf.add_virtual_loss(up_to=self.root)
                move_prob, t_value = self.nnet.predict(self.get_cum_view(cur_node))
                cur_node.incorporate_results(move_prob, t_value, up_to=self.mctc_root)

    def play_game(self, render=False):
        self.init_game()
        node = self.mctc_root
        step = 0
        train_examples = []
        view_queue = collections.deque(maxlen=8)
        view_queue.append(node.view)
        while True:
            step += 1
            action = self.pick_move(node, step)
            pi, q, next_node, view = self.play_move(node, step, action)
            node = next_node
            view_queue.append(node.view)
            future_view = self.get_cum_view_from_queue(view_queue.copy())
            train_examples.append([future_view, pi])
            if render:
                self.game.env.render()
            if self.game.isDone():
                value = np.tanh(self.game.getTotalScore())
                return [(x[0],x[1], value) for x in train_examples]


    def play_move(self, node, step, action):
        pi = node.children_as_pi(step < TEMPERATURE_CUTOFF)
        q = node.Q
        next_node, reward, view = node.maybe_add_child(action, self.game)
        return pi,q, next_node, view


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


    def get_cum_view(self, node):
        result = first =  node.view
        shape = result.shape
        leaf = node
        for x in range(7):
            if leaf.parent.parent is not None:
                result = np.dstack((result, leaf.parent.view))
                leaf = leaf.parent
            else:
                result = np.dstack((first, result))
        return result

    def get_cum_view_from_queue(self, queue):
        result = first = queue.popleft()
        shape = result.shape
        for x in range(7):
            if len(queue) > 0:
                result = np.dstack((result, queue.popleft()))
            else:
                result = np.dstack((first, result))
        return result