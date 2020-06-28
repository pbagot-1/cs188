# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        penalty = 0
        if action == Directions.STOP:
            penalty = 20
        nearestGhostManhattan = min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]) if [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in   newGhostStates] else 0
        distanceFromCenter = manhattanDistance(newPos, (currentGameState.data.layout.width/2, currentGameState.data.layout.height/2))
        nearestFood = min([manhattanDistance(newPos, foodSpot) for foodSpot in newFood.asList()]) if [manhattanDistance(newPos, foodSpot) for foodSpot in newFood.asList()] else 0
        return 10*successorGameState.getScore() + 1/(len(newFood.asList()) + 1) + nearestGhostManhattan + 200*(1/(nearestFood + 1)) + 10*(1/distanceFromCenter) - penalty

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

   
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentNum):
        Returns a list of legal actions for an agent
        agentNum=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentNum, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def evalMaximizer(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("-inf")
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                v = max(v, evalMinimizer(gameState.generateSuccessor(0, action), depth, 1))
            return v
        
        def evalMinimizer(gameState, depth, agentNum):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("inf")
            legalActions = gameState.getLegalActions(agentNum)
            if agentNum == gameState.getNumAgents() - 1:
                for action in legalActions:
                    v = min(v, evalMaximizer(gameState.generateSuccessor(agentNum, action), depth - 1))
            else:
                for action in legalActions:
                    v = min(v, evalMinimizer(gameState.generateSuccessor(agentNum, action), depth, agentNum + 1))
            return v
        depth = self.depth
        minimaxAction = Directions.STOP
        maxval = max([evalMinimizer(gameState.generateSuccessor(0, action), depth, 1) for action in gameState.getLegalActions(0)])
        index = [evalMinimizer(gameState.generateSuccessor(0, action), depth,  1) for action in gameState.getLegalActions(0)].index(maxval)
        minimaxAction = [action for action in gameState.getLegalActions(0)][index]
    
        
       
        
        return minimaxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def evalMaximizer(gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("-inf")
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                v = max(v, evalMinimizer(gameState.generateSuccessor(0, action), depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        def evalMinimizer(gameState, depth, agentNum, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("inf")
            legalActions = gameState.getLegalActions(agentNum)
            if agentNum == gameState.getNumAgents() - 1:
                for action in legalActions:
                    v = min(v, evalMaximizer(gameState.generateSuccessor(agentNum, action), depth - 1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            else:
                for action in legalActions:
                    v = min(v, evalMinimizer(gameState.generateSuccessor(agentNum, action), depth, agentNum + 1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            
            return v
        depth = self.depth
        minimaxAction = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")
        """maxval = max([evalMinimizer(gameState.generateSuccessor(0, action), depth, 1, alpha, beta) for action in gameState.getLegalActions(0)])
        index = [evalMinimizer(gameState.generateSuccessor(0, action), depth,  1, alpha, beta) for action in gameState.getLegalActions(0)].index(maxval)
        minimaxAction = [action for action in gameState.getLegalActions(0)][index]"""
        score = float("-inf")
        for action in gameState.getLegalActions(0):
            hold = score
            score = max(score, evalMinimizer(gameState.generateSuccessor(0, action), depth, 1, alpha, beta))
            if score > hold:
                minimaxAction = action
            alpha = max(alpha, score)
        
       
        
        return minimaxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def evalMaximizer(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("-inf")
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                v = max(v, evalMinimizer(gameState.generateSuccessor(0, action), depth, 1))
            return v
        
        def evalMinimizer(gameState, depth, agentNum):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            #v = float("inf")
            legalActions = gameState.getLegalActions(agentNum)
            numToDivideBy = len(legalActions)
            utilityForEach = 0
            if agentNum == gameState.getNumAgents() - 1:
                for action in legalActions:
                    utilityForEach += evalMaximizer(gameState.generateSuccessor(agentNum, action), depth - 1)
            else:
                for action in legalActions:
                    utilityForEach += evalMinimizer(gameState.generateSuccessor(agentNum, action), depth, agentNum + 1)
            return utilityForEach/numToDivideBy
        depth = self.depth
        minimaxAction = Directions.STOP
        maxval = max([evalMinimizer(gameState.generateSuccessor(0, action), depth, 1) for action in gameState.getLegalActions(0)])
        index = [evalMinimizer(gameState.generateSuccessor(0, action), depth,  1) for action in gameState.getLegalActions(0)].index(maxval)
        minimaxAction = [action for action in gameState.getLegalActions(0)][index]
        return minimaxAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    For the first term, -6*nearestFood, I wanted the eval function to punish being further away from food, because that will take longer for a winning state.
    For second term, nearestGhostManhattan, I wanted it to make it better if the ghost was further away, which maybe is not necessarily the best way to do it, since for example the ghost being 5 vs 50 away 
    doesn't make too much of a difference.
    For third term 10*currentGameState.getScore(), I wanted a strong incentive for getting a higher score. 
    For fourth term -5*len(food), I wanted the food list being longer to hurt pacmans score. 
    For fifth term, -20*len(capsules), I wanted it to encourage picking up capsules when near them, because it helps with increasing score by eating ghosts.
    """
    """if currentGameState.isWin():
       return float("inf")
    if currentGameState.isLose():
        return float ("-inf")"""
        
    nearestGhostManhattan = min([manhattanDistance(currentGameState.getPacmanPosition(), ghostState.getPosition()) for ghostState in currentGameState.getGhostStates()]) if [manhattanDistance(currentGameState.getPacmanPosition(), ghostState.getPosition()) for ghostState in currentGameState.getGhostStates()] else 0
    nearestCapsule = min([manhattanDistance(currentGameState.getPacmanPosition(), capsule) for capsule in currentGameState.getCapsules()]) if [manhattanDistance(currentGameState.getPacmanPosition(), capsule) for capsule in currentGameState.getCapsules()] else 0
    nearestFood = min([manhattanDistance(currentGameState.getPacmanPosition(), foodSpot) for foodSpot in currentGameState.getFood().asList()]) if [manhattanDistance(currentGameState.getPacmanPosition(), foodSpot) for foodSpot in currentGameState.getFood().asList()] else 0
    return -6*nearestFood + nearestGhostManhattan + 10*currentGameState.getScore() - 5*len(currentGameState.getFood().asList()) - 20*len(currentGameState.getCapsules())
    


# Abbreviation
better = betterEvaluationFunction
