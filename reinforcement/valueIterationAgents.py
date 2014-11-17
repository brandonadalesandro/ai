# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        
        "*** YOUR CODE HERE ***"
        # each iteration, make a copy
        # find the max of each state, action pair for each state
        # store it in self.values
        # adapted from pseudocode from the sides, and info found on wikipedia (http://en.wikipedia.org/wiki/Markov_decision_process)
        #
        for i in range(iterations):
            # initially I was just storing the values,
            # but I realized that I was sometimes updating 
            # from within the same state, a very long standing bug in my code
            values = self.values.copy()
            for state in mdp.getStates():
                # decided to skip over terminal states, weird interactions with the autograder
                if self.mdp.isTerminal(state):
                    continue
                # Use a list and take the max to avoid the if None or val > max_val loop
                q_values = []
                for action in mdp.getPossibleActions(state):
                    val = self.computeQValueFromValues(state, action)
                    q_values.append(val)
                values[state] = max(q_values)
            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # given a state, action pair, get all the necessary components to calculate the Q value
        # Q = T * (R + distcount * V[k - 1])
        ret_val = 0
       	for next_state, next_prob in self.mdp.getTransitionStatesAndProbs(state, action):
       		ret_val += next_prob * (self.mdp.getReward(state, action, next_state) + (self.discount * self.values[next_state]))
       	return ret_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_val = None
        max_act = None
        # for each possible action, get the max comb of state, action
        for action in self.mdp.getPossibleActions(state):
            val = self.computeQValueFromValues(state, action)
            if max_val is None or val > max_val:
                max_val = val
                max_act = action
        return max_act

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
