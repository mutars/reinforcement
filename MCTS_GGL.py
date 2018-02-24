# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monte Carlo Tree Search implementation.

All terminology here (Q, U, N, p_UCT) uses the same notation as in the
AlphaGo (AG) paper.
"""

import numpy as np
import collections
import random
import math

# import coords
# import go

MAX_DEPTH = 1000000

# Exploration constant
c_PUCT = 1.38


# Dirichlet noise, as a function of go.N


class DummyNode(object):
    """A fake node of a MCTS search tree.

    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class MCTSNode(object):
    """A node of a MCTS search tree.

    A node knows how to compute the action scores of all of its children,
    so that a decision can be made about which move to explore next. Upon
    selecting a move, the children dictionary is updated with a new node.

    position: A go.Position instance
    fmove: A move (coordinate) that led to this position, a a flattened coord
            (raw number between 0-N^2, with None a pass)
    parent: A parent MCTSNode.
    """

    def __init__(self, view, action_size, action=None, parent=None):
        if parent is None:
            parent = DummyNode()
        self.parent = parent
        self.action = action  # move that led to this position, as flattened coords
        self.view = view
        self.is_expanded = False
        self.action_size = action_size

        #self.child_short_term_v = np.zeros([action_size], dtype=np.float32)
        #self.child_short_term_n = np.zeros([action_size], dtype=np.float32)
        self.child_N = np.zeros([action_size], dtype=np.float32)
        self.child_W = np.zeros([action_size], dtype=np.float32)
        # save a copy of the original prior before it gets mutated by d-noise.
        self.original_prior = np.zeros([action_size], dtype=np.float32)
        self.child_prior = np.zeros([action_size], dtype=np.float32)
        self.children = {}  # map of flattened moves to resulting MCTSNode

    def __repr__(self):
        # TODO
        return "Node =%s" % self.N

    @property
    def child_action_score(self):
        return self.child_Q + self.child_U

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (c_PUCT * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

    @property
    def Q(self):
        return self.W / (1 + self.N)

    @property
    def N(self):
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value

    ''''@property
    def short_W(self):
        return self.parent.child_short_term_v[self.action]

    @short_W.setter
    def short_W(self, value):
        self.parent.child_short_term_v[self.action] = value
        self.parent.child_short_term_n[self.action] += 1'''

    def select_leaf(self, game):
        current = self
        reward = 0
        # pass_move = self.action_size * self.action_size
        while True:
            current.N += 1
            # if a node has never been evaluated, we have no basis to select a child.
            if not current.is_expanded:
                break

            best_action = np.argmax(current.child_action_score)
            current, reward, new_view = current.maybe_add_child(best_action, game)
        return current, reward

    def maybe_add_child(self, action, game):
        """ Adds child node for fcoord if it doesn't already exist, and returns it. """
        reward = 0
        if action not in self.children:
            new_view, reward, _ = game.getNextState(action)
            self.children[action] = MCTSNode(
                new_view, action_size=self.action_size, action=action, parent=self)
        else:
            new_view, reward, _ = game.getNextState(action)
            #should be equal to
            #assert self.children[action].view == new_view
        return self.children[action], reward, new_view

    def revert_visits(self, up_to):
        """Revert visit increments.

        Sometimes, repeated calls to select_leaf return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But select_leaf has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.
        """
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)



    def incorporate_results(self, move_probabilities, total_value,  eventual_value =0, up_to=None):
        assert move_probabilities.shape == (self.action_size,)
        # A finished game should not be going through this code path - should
        # directly call backup_value() on the result of the game.
        # TODO or not todo
        # assert not self.view.is_game_over()
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = move_probabilities
        # initialize child Q as current node's value, to prevent dynamics where
        # if B is winning, then B will only ever explore 1 move, because the Q
        # estimation will be so much larger than the 0 of the other moves.
        #
        # Conversely, if W is winning, then B will explore all 362 moves before
        # continuing to explore the most favorable move. This is a waste of search.
        #
        # The value seeded here acts as a prior, and gets averaged into Q calculations.
        self.child_W = np.ones([self.action_size], dtype=np.float32) * total_value
        self.backup_total_value(total_value, up_to=up_to)

    def backup_total_value(self, value, up_to=None):
        """Propagates a value estimation up to the root node.

        Args:
            value: the value to be propagated (1 = black wins, -1 = white wins)
            up_to: the node to propagate until.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_total_value(value, up_to)

    def D_NOISE_ALPHA(self):
        return 0.03 * 5 / (self.action_size)

    def inject_noise(self):
        dirch = np.random.dirichlet([self.D_NOISE_ALPHA()] * (self.action_size + 1))
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def children_as_pi(self, squash=False):
        """Returns the child visit counts as a probability distribution, pi
        If squash is true, exponentiate the probabilities by a temperature
        slightly larger than unity to encourage diversity in early play and
        hopefully to move away from 3-3s
        """
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)

    def most_visited_path(self):
        node = self
        output = []
        while node.children:
            next_kid = np.argmax(node.child_N)
            node = node.children.get(next_kid)
            if node is None:
                output.append("GAME END")
                break
            """output.append("%s (%d) ==> " % (coords.to_kgs(
                coords.from_flat(node.fmove)),
                                            node.N))"""
        output.append("Q: {:.5f}\n".format(node.Q))
        return ''.join(output)

    def mvp_gg(self):
        """ Returns most visited path in go-gui VAR format e.g. 'b r3 w c17..."""
        node = self
        output = []
        while node.children and max(node.child_N) > 1:
            next_kid = np.argmax(node.child_N)
            node = node.children[next_kid]
            """output.append("%s" % coords.to_kgs(
                coords.from_flat(node.fmove)))"""
        return ' '.join(output)

    def describe(self):
        sort_order = list(range(self.action_size * self.action_size + 1))
        sort_order.sort(key=lambda i: (
            self.child_N[i], self.child_action_score[i]), reverse=True)
        soft_n = self.child_N / sum(self.child_N)
        p_delta = soft_n - self.child_prior
        p_rel = p_delta / self.child_prior
        # Dump out some statistics
        output = []
        output.append("{q:.4f}\n".format(q=self.Q))
        output.append(self.most_visited_path())
        output.append(
            "move:  action      Q      U      P    P-Dir    N  soft-N  p-delta  p-rel\n")
        output.append("\n".join(["{!s:6}: {: .3f}, {: .3f}, {:.3f}, {:.3f}, {:.3f}, {:4d} {:.4f} {: .5f} {: .2f}".format(
            self.child_action_score[key],
            self.child_Q[key],
            self.child_U[key],
            self.child_prior[key],
            self.original_prior[key],
            int(self.child_N[key]),
            soft_n[key],
            p_delta[key],
            p_rel[key])
                                    for key in sort_order][:15]))
        return ''.join(output)
