# Sample MDPs for Advanced AI

This contains a set of MDPs to test your code on. They are all fairly
simple, so reading the code should help explain most of it.

The only functions needed for a policy learning/q-learning agent are
given in `mdp.py`. There are three different domains:

* Discrete Grid World in `gridworld.py`: An agent can move up, down,
  left and right with some noise, and must avoid pits to reach some
  goal.

* Continuous Grid World in `gridworld_c.py`: Continuous variant of the
  above.

* Wumpus domain: Similar to discrete grid world, but wumpuses kill and
  pits only cause some damage. The agent can pick up gold for extra
  reward or immunity to pass through wumpuses.

There are also some sample simulations of each domain given in the
`-sample.py` files.