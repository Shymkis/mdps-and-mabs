from gridworld_c import ContinuousGridWorldMDP
import numpy as np


mdp = ContinuousGridWorldMDP(50, 50)

mdp.add_pit(5, 5, 1)
mdp.add_pit(5, 10, 1)
mdp.add_pit(10, 20, 1)
mdp.add_pit(40, 41, 2)

mdp.add_goal(50, 50, 5)

x = mdp.initial_state
t = 0

while not mdp.is_terminal(x) and t < 1000:
    print(x)
    a = np.random.choice(list(mdp.actions_at(x)))
    print(a)
    x, _ = mdp.act(x, a)
    t += 1

print(x)
