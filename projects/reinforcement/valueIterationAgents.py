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
import collections

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
                    
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.

        Note that in Q-Learning and reinforcment
        learning in general, we do not know these
        probabilities nor do we directly model them.
       
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for state in self.mdp.getStates():
            self.values[state] = 0
        k = 0
        while (k != self.iterations):
            k = k + 1
            valuesDup = util.Counter() 
            for state in self.mdp.getStates():
                listOfVals = []
                for action in self.mdp.getPossibleActions(state):
                    sum = 0
                    for pair in self.mdp.getTransitionStatesAndProbs(state, action):
                        sum = sum + (pair[1]*(self.mdp.getReward(state, action, pair[0]) + self.discount*(self.values[pair[0]])))
                    listOfVals.append(sum)
                valuesDup[state] = max(listOfVals) if len(listOfVals) != 0 else 0
            #set values to new "batch updated" dictionary
            self.values = valuesDup
        


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
        sum = 0
        for pair in self.mdp.getTransitionStatesAndProbs(state, action):
           sum = sum + (pair[1]*(self.mdp.getReward(state, action, pair[0]) + self.discount*(self.values[pair[0]])))
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actionsList = self.mdp.getPossibleActions(state)
        if len(actionsList) == 0:
            return None
        #index = 0
        listOfVals = []
        for action in actionsList:
            sum = 0
            for pair in self.mdp.getTransitionStatesAndProbs(state, action):
                sum = sum + (pair[1]*(self.mdp.getReward(state, action, pair[0]) + self.discount*(self.values[pair[0]])))
                
            listOfVals.append(sum)
        
        return actionsList[listOfVals.index(max(listOfVals))]
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
    # Write value iteration code here
        for state in self.mdp.getStates():
            self.values[state] = 0
        k = 0
        while (k != self.iterations):
           
            #valuesDup = util.Counter() 
            for state in self.mdp.getStates():
                if (k == self.iterations):
                    break
                k = k + 1
                if self.mdp.isTerminal(state) is False:
                    listOfVals = []
                    for action in self.mdp.getPossibleActions(state):
                        sum = 0
                        for pair in self.mdp.getTransitionStatesAndProbs(state, action):
                            sum = sum + (pair[1]*(self.mdp.getReward(state, action, pair[0]) + self.discount*(self.values[pair[0]])))
                        listOfVals.append(sum)
                    self.values[state] = max(listOfVals) if len(listOfVals) != 0 else 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
            self.values[state] = 0
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for pair in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[pair[0]].add(state)
                    
        priorityQueue = util.PriorityQueue()
        
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s) is False:
                listOfVals = []
                for action in self.mdp.getPossibleActions(s):
                    sum = 0
                    for pair in self.mdp.getTransitionStatesAndProbs(s, action):
                        sum = sum + (pair[1]*(self.mdp.getReward(s, action, pair[0]) + self.discount*(self.values[pair[0]])))
                    listOfVals.append(sum)
                highestQVal = max(listOfVals) if len(listOfVals) != 0 else 0
                diff = abs(self.values[s] - highestQVal)
                priorityQueue.push(s, diff*-1)
        for iteration in range(0, self.iterations):
            if priorityQueue.isEmpty():
                break
            s = priorityQueue.pop()
            listOfVals = []
            for action in self.mdp.getPossibleActions(s):
                sum = 0
                for pair in self.mdp.getTransitionStatesAndProbs(s, action):
                    sum = sum + (pair[1]*(self.mdp.getReward(s, action, pair[0]) + self.discount*(self.values[pair[0]])))
                listOfVals.append(sum)
            self.values[s] = max(listOfVals) if len(listOfVals) != 0 else 0
            for p in predecessors[s]:
                if self.mdp.isTerminal(p) is False:
                    listOfVals = []
                    for action in self.mdp.getPossibleActions(p):
                        sum = 0
                        for pair in self.mdp.getTransitionStatesAndProbs(p, action):
                            sum = sum + (pair[1]*(self.mdp.getReward(p, action, pair[0]) + self.discount*(self.values[pair[0]])))
                        listOfVals.append(sum)
                    highestQVal = max(listOfVals) if len(listOfVals) != 0 else 0
                    diff = abs(self.values[p] - highestQVal)
                    if diff > self.theta:
                        priorityQueue.update(p, diff*-1)
                
            

