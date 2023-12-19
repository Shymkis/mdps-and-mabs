from gridworld import DiscreteGridWorldMDP, _TXT
from wumpus import WumpusMDP, _TXT as W_TXT
from time import time
import numpy as np
import random


def value_iteration(mdp, gamma: float = .9, tol: float = 1e-6):
    # 1. Initialization
    wumpus_world = isinstance(mdp, WumpusMDP)
    z = (mdp._w, mdp._h, 2, 2) if wumpus_world else (mdp._w, mdp._h)
    v = np.zeros(z).tolist()
    pi = np.zeros_like(v).tolist()
    # 2. Value Iteration
    Delta = tol
    while not Delta < tol:
        Delta = 0
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue
            tmp = v[s.x][s.y][s.has_gold][s.has_immunity] if wumpus_world else v[s.x][s.y]
            totals = []
            for a in mdp.actions_at(s):
                pairs = mdp.pick_up(s, a) if a == mdp.actions.PICK_UP else mdp.move(s, a)
                totals.append(0)
                for s2, p in pairs:
                    val = v[s2.x][s2.y][s2.has_gold][s2.has_immunity] if wumpus_world else v[s2.x][s2.y]
                    totals[-1] += p*(mdp.r(s, s2) + gamma*val)
            if wumpus_world:
                v[s.x][s.y][s.has_gold][s.has_immunity] = max(totals)
            else:
                v[s.x][s.y] = max(totals)
            diff = abs(tmp - v[s.x][s.y][s.has_gold][s.has_immunity]) if wumpus_world else abs(tmp - v[s.x][s.y])
            Delta = max(Delta, diff)
    # 3. Policy Extraction
    for s in mdp.states:
        if mdp.is_terminal(s):
            continue
        totals = {}
        for a in mdp.actions_at(s):
            pairs = mdp.pick_up(s, a) if a == mdp.actions.PICK_UP else mdp.move(s, a)
            totals[a] = 0
            for s2, p in pairs:
                val = v[s2.x][s2.y][s2.has_gold][s2.has_immunity] if wumpus_world else v[s2.x][s2.y]
                totals[a] += p*(mdp.r(s, s2) + gamma*val)
        if wumpus_world:
            pi[s.x][s.y][s.has_gold][s.has_immunity] = max(totals, key=lambda x: totals[x])
        else:
            pi[s.x][s.y] = max(totals, key=lambda x: totals[x])
    return v, pi

def policy_iteration(mdp, gamma: float = .9, tol: float = 1e-6, max_it: int = 100):
    """Policy Iteration algorithm.

    Args:
        mdp (MDP): The MDP to solve.
        gamma (float, optional): The discount rate. Defaults to .9.
        tol (float, optional): The minimum tolerance to detect a change. Defaults to 1e-6.
        max_it (int, optional): The maximum number of policy iterations. Defaults to 100.

    Returns:
        v (list): The maximal values at each state.
        pi (list): The optimal actions at each state.
    """
    # 1. Initialization
    wumpus_world = isinstance(mdp, WumpusMDP)
    z = (mdp._w, mdp._h, 2, 2) if wumpus_world else (mdp._w, mdp._h)
    v = np.zeros(z).tolist()
    pi = np.zeros_like(v).tolist()
    # 2. Policy Iteration
    for _ in range(max_it):
        # 2.a. Policy Evaluation
        Delta = tol
        while not Delta < tol:
            Delta = 0
            for s in mdp.states:
                if mdp.is_terminal(s):
                    continue
                tmp = v[s.x][s.y][s.has_gold][s.has_immunity] if wumpus_world else v[s.x][s.y]
                if wumpus_world:
                    if pi[s.x][s.y][s.has_gold][s.has_immunity] == 0:
                        pi[s.x][s.y][s.has_gold][s.has_immunity] = np.random.choice(mdp.actions_at(s))
                    a = pi[s.x][s.y][s.has_gold][s.has_immunity]
                else:
                    if pi[s.x][s.y] == 0:
                        pi[s.x][s.y] = np.random.choice(mdp.actions_at(s))
                    a = pi[s.x][s.y]
                pairs = mdp.pick_up(s, a) if a == mdp.actions.PICK_UP else mdp.move(s, a)
                total = 0
                for s2, p in pairs:
                    val = v[s2.x][s2.y][s2.has_gold][s2.has_immunity] if wumpus_world else v[s2.x][s2.y]
                    total += p*(mdp.r(s, s2) + gamma*val) # Q-value, T(s,a,s')[R(s,a,s') + gamma*U(s')]
                if wumpus_world:
                    v[s.x][s.y][s.has_gold][s.has_immunity] = total
                else:
                    v[s.x][s.y] = total
                diff = abs(tmp - v[s.x][s.y][s.has_gold][s.has_immunity]) if wumpus_world else abs(tmp - v[s.x][s.y])
                Delta = max(Delta, diff)
        # 2.b. Policy Improvement
        stable = True
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue
            tmp = pi[s.x][s.y][s.has_gold][s.has_immunity] if wumpus_world else pi[s.x][s.y]
            totals = {}
            for a in mdp.actions_at(s):
                pairs = mdp.pick_up(s, a) if a == mdp.actions.PICK_UP else mdp.move(s, a)
                totals[a] = 0
                for s2, p in pairs:
                    val = v[s2.x][s2.y][s2.has_gold][s2.has_immunity] if wumpus_world else v[s2.x][s2.y]
                    totals[a] += p*(mdp.r(s, s2) + gamma*val)
            if wumpus_world:
                pi[s.x][s.y][s.has_gold][s.has_immunity] = max(totals, key=lambda x: totals[x])
            else:
                pi[s.x][s.y] = max(totals, key=lambda x: totals[x])
            if wumpus_world:
                if tmp is not pi[s.x][s.y][s.has_gold][s.has_immunity]:
                    stable = False
            else:
                if tmp is not pi[s.x][s.y]:
                    stable = False
        if stable:
            break
    return v, pi

def epsilon_greedy(mdp, q, s, eps: float = .1):
    """Action selection using the epsilon-greedy algorithm.

    Args:
        mdp (MDP): The given MDP.
        q (np.ndarray): Q-table with the structure q[x][y][a].
        s (state): The current state.
        eps (float, optional): The epsilon value. Defaults to .1.

    Returns:
        action: Maximal action given Q-table and state, or random action.
    """
    if random.random() < eps:
        return np.random.choice(mdp.actions)
    if isinstance(mdp, WumpusMDP):
        return mdp.actions(q[s.x][s.y][int(s.has_gold)][int(s.has_immunity)].argmax()+1)
    return mdp.actions(q[s.x][s.y].argmax()+1)

def q_learning(mdp, gamma: float = .9, max_it: int = 5000):
    """Q-learning algorithm.

    Args:
        mdp (MDP): The given MDP.
        gamma (float, optional): The discount factor. Defaults to .9.
        max_it (int, optional): The number of episodes to run. Defaults to 5000.

    Returns:
        q: Q-table with the structure q[x][y][a].
    """
    if isinstance(mdp, WumpusMDP):
        q = np.zeros((mdp._w, mdp._h, 2, 2, len(mdp.actions)))
        for t in range(max_it):
            print(t)
            alpha = epsilon = 1/(t+1)
            s = mdp.initial_state
            while not mdp.is_terminal(s): # Complete an episode
                a = epsilon_greedy(mdp, q, s, epsilon)
                s2, r = mdp.act(s, a)
                q[s.x][s.y][int(s.has_gold)][int(s.has_immunity)][a.value-1] += alpha*(r + gamma*max(q[s2.x][s2.y][int(s2.has_gold)][int(s2.has_immunity)]) - q[s.x][s.y][int(s.has_gold)][int(s.has_immunity)][a.value-1])
                s = s2
    else:
        q = np.zeros((mdp._w, mdp._h, len(mdp.actions)))
        for t in range(max_it):
            alpha = epsilon = 1/(t+1)
            s = mdp.initial_state
            while not mdp.is_terminal(s): # Complete an episode
                a = epsilon_greedy(mdp, q, s, epsilon)
                s2, r = mdp.act(s, a)
                q[s.x][s.y][a.value-1] += alpha*(r + gamma*max(q[s2.x][s2.y]) - q[s.x][s.y][a.value-1])
                s = s2
    return q

def sarsa(mdp, gamma: float = .9, max_it: int = 5000):
    """SARSA algorithm.

    Args:
        mdp (MDP): The given MDP.
        gamma (float, optional): The discount factor. Defaults to .9.
        max_it (int, optional): The number of episodes to run. Defaults to 5000.

    Returns:
        q: Q-table with the structure q[x][y][a].
    """
    if isinstance(mdp, WumpusMDP):
        q = np.zeros((mdp._w, mdp._h, 2, 2, len(mdp.actions)))
        for t in range(max_it):
            print(t)
            alpha = epsilon = 1/(t+1)
            s = mdp.initial_state
            a = epsilon_greedy(mdp, q, s, epsilon)
            while not mdp.is_terminal(s): # Complete an episode
                s2, r = mdp.act(s, a)
                a2 = epsilon_greedy(mdp, q, s2, epsilon)
                q[s.x][s.y][int(s.has_gold)][int(s.has_immunity)][a.value-1] += alpha*(r + gamma*q[s2.x][s2.y][int(s.has_gold)][int(s.has_immunity)][a2.value-1] - q[s.x][s.y][int(s.has_gold)][int(s.has_immunity)][a.value-1])
                s, a = s2, a2
    else:
        q = np.zeros((mdp._w, mdp._h, len(mdp.actions)))
        for t in range(max_it):
            alpha = epsilon = 1/(t+1)
            s = mdp.initial_state
            a = epsilon_greedy(mdp, q, s, epsilon)
            while not mdp.is_terminal(s): # Complete an episode
                s2, r = mdp.act(s, a)
                a2 = epsilon_greedy(mdp, q, s2, epsilon)
                q[s.x][s.y][a.value-1] += alpha*(r + gamma*q[s2.x][s2.y][a2.value-1] - q[s.x][s.y][a.value-1])
                s, a = s2, a2
    return q

def convert_actions_to_symbols(pi, mdp):
    if isinstance(mdp, WumpusMDP):
        for x in range(len(pi)):
            for y in range(len(pi[x])):
                for has_gold in range(len(pi[x][y])):
                    for has_immunity in range(len(pi[x][y][has_gold])):
                        if mdp.obs_at("goal", (x,y)):
                            pi[x][y][has_gold][has_immunity] = " X"
                        elif mdp.obs_at("pit", (x,y)):
                            pi[x][y][has_gold][has_immunity] = " P"
                        elif mdp.obs_at("wumpus", (x,y)):
                            pi[x][y][has_gold][has_immunity] = " W"
                        else:
                            if isinstance(pi[x][y][has_gold][has_immunity], mdp.actions):
                                pi[x][y][has_gold][has_immunity] = W_TXT[pi[x][y][has_gold][has_immunity]]
                            else:
                                pi[x][y][has_gold][has_immunity] = W_TXT[mdp.actions(pi[x][y][has_gold][has_immunity] + 1)]
    else:
        for x in range(len(pi)):
            for y in range(len(pi[x])):
                if mdp.obs_at("goal", (x,y)):
                    pi[x][y] = " X"
                elif mdp.obs_at("pit", (x,y)):
                    pi[x][y] = " P"
                else:
                    if isinstance(pi[x][y], mdp.actions):
                        pi[x][y] = _TXT[pi[x][y]]
                    else:
                        print(pi[x][y])
                        pi[x][y] = _TXT[mdp.actions(pi[x][y])]


if __name__ == "__main__":
    # Create MDP
    # 4x3 grid
    # mdp = DiscreteGridWorldMDP(4, 3, -0.04)
    # mdp.add_obstacle("goal", (3,2))
    # mdp.add_obstacle("pit", (3,1))
    # mdp.add_obstacle("pit", (1,1))
    # 10x10 grid 1
    # mdp = DiscreteGridWorldMDP(10, 10, -0.04)
    # mdp.add_obstacle("goal", (9,9))
    # 10x10 grid 2
    mdp = DiscreteGridWorldMDP(10, 10, -0.04)
    mdp.add_obstacle("goal", (4,4))
    # Wumpus world
    # mdp = WumpusMDP(8, 10, -0.04)
    # mdp.add_obstacle('wumpus', [6, 7])
    # mdp.add_obstacle('wumpus', [6, 8])
    # mdp.add_obstacle('wumpus', [6, 9], -100)
    # mdp.add_obstacle('wumpus', [7, 5])
    # mdp.add_obstacle('pit', [2, 0])
    # mdp.add_obstacle('pit', [2, 1])
    # mdp.add_obstacle('pit', [2, 2], -0.5)
    # mdp.add_obstacle('pit', [5, 0])
    # mdp.add_obstacle('pit', [6, 1])
    # mdp.add_obstacle('goal', [7, 9])
    # mdp.add_object('gold', [0, 9])
    # mdp.add_object('gold', [1, 1])
    # mdp.add_object('gold', [7, 0])
    # mdp.add_object('immune', [6, 0])
    # mdp.add_object('immune', [1, 2])
    # mdp.display()

    # Obtain values and policies
    # v_v, pi_v = value_iteration(mdp)
    # v_p, pi_p = policy_iteration(mdp)

    q_q = q_learning(mdp)
    v_q = q_q.max(axis=2).tolist()
    pi_q = (q_q.argmax(axis=2)+1).tolist()

    q_s = sarsa(mdp)
    v_s = q_s.max(axis=2).tolist()
    pi_s = (q_s.argmax(axis=2)+1).tolist()

    # Convert actions to symbols
    # convert_actions_to_symbols(pi_v, mdp)
    # convert_actions_to_symbols(pi_p, mdp)
    convert_actions_to_symbols(pi_q, mdp)
    convert_actions_to_symbols(pi_s, mdp)

    # Display policy values and actions
    # print("Value Iteration")
    # if isinstance(mdp, WumpusMDP):
    #     print("No gold, no immunity")
    #     print(np.rot90(np.array(pi_v)[:,:,0,0]))
    #     print("No gold, has immunity")
    #     print(np.rot90(np.array(pi_v)[:,:,0,1]))
    #     print("Has gold, no immunity")
    #     print(np.rot90(np.array(pi_v)[:,:,1,0]))
    #     print("Has gold, has immunity")
    #     print(np.rot90(np.array(pi_v)[:,:,1,1]))
    # else:
    #     print(np.rot90(v_v))
    #     print(np.rot90(pi_v))
    # print("Policy Iteration")
    # if isinstance(mdp, WumpusMDP):
    #     print("No gold, no immunity")
    #     print(np.rot90(np.array(pi_p)[:,:,0,0]))
    #     print("No gold, has immunity")
    #     print(np.rot90(np.array(pi_p)[:,:,0,1]))
    #     print("Has gold, no immunity")
    #     print(np.rot90(np.array(pi_p)[:,:,1,0]))
    #     print("Has gold, has immunity")
    #     print(np.rot90(np.array(pi_p)[:,:,1,1]))
    # else:
    #     print(np.rot90(v_p))
    #     print(np.rot90(pi_p))
    # print("Q-Learning")
    if isinstance(mdp, WumpusMDP):
        print("No gold, no immunity")
        print(np.rot90(np.array(pi_q)[:,:,0,0]))
        print("No gold, has immunity")
        print(np.rot90(np.array(pi_q)[:,:,0,1]))
        print("Has gold, no immunity")
        print(np.rot90(np.array(pi_q)[:,:,1,0]))
        print("Has gold, has immunity")
        print(np.rot90(np.array(pi_q)[:,:,1,1]))
    else:
        print(np.rot90(v_q))
        print(np.rot90(pi_q))
    print("SARSA")
    if isinstance(mdp, WumpusMDP):
        print("No gold, no immunity")
        print(np.rot90(np.array(pi_s)[:,:,0,0]))
        print("No gold, has immunity")
        print(np.rot90(np.array(pi_s)[:,:,0,1]))
        print("Has gold, no immunity")
        print(np.rot90(np.array(pi_s)[:,:,1,0]))
        print("Has gold, has immunity")
        print(np.rot90(np.array(pi_s)[:,:,1,1]))
    else:
        print(np.rot90(v_s))
        print(np.rot90(pi_s))
