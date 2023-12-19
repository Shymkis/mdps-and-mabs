#!/usr/bin/env python3

################################################################
# MDP.PY
#
# Specifies all functionality needed for the MDP. Any agent should
# only call functions specified in this file.
################################################################

import numpy as np


class MDPState:
    """A single state within the MDP."""
    @property
    def i(self):
        """Returns a unique index associated with each state (useful for q table)."""
        pass


class MDP:
    """MDP specification and simulator."""
    @property
    def actions(self):
        """Return iterable of all actions."""
        raise NotImplementedError

    def actions_at(self, state):
        """Return iterable of all actions at given state."""
        raise NotImplementedError

    @property
    def initial_state(self):
        """Returns initial state (assumed determinstic)."""
        raise NotImplementedError
    
    def r(self, s1, s2):
        """Returns the reward for transitioning from s1 to s2. For now, assume it is deterministic."""
        raise NotImplementedError

    def is_terminal(self, state):
        """Returns true if state s is terminal."""
        raise NotImplementedError

    def act(self, state, action):
        """Observe a single MDP transition."""
        raise NotImplementedError

class FiniteStateMDP:
    """MDP with finite states."""
    @property
    def num_states(self):
        """Returns the number of states."""
        raise NotImplementedError

    @property
    def states(self):
        """Return iterable of all states. (if discrete)"""
        raise NotImplementedError

    def p(self, state, action):
        """Return an iterable of state-probability pairs when performing action in state."""
        raise NotImplementedError

    def act(self, state, action):
        x, p = zip(*self.p(state, action))
        s2 = np.random.choice(x, p=p)
        return s2, self.r(state, s2)
