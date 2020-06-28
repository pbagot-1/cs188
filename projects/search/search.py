 # search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    #print("Start:", problem.getStartState())
   # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    "*** YOUR CODE HERE ***"
    actionList = []
    currentState = (problem.getStartState(), 0 , 0)
    closed = set([])
    #each state maps to its parent
    #moveDict = {}
    if problem.isGoalState(problem.getStartState()):
        return []
    successorStack = util.Stack()
    successorStack.push((problem.getStartState(), []))

    while not successorStack.isEmpty():
        currentState = successorStack.pop()
        """if (currentState[1] == 'North'):
            moveDict[currentState[0]] = ((currentState[0][0] , currentState[0][1] - 1), 'North')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] , currentState[0][1] - 1)))
        if (currentState[1] == 'South'):
            moveDict[currentState[0]] = ((currentState[0][0] , currentState[0][1] + 1), 'South')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] , currentState[0][1] + 1)))
        if (currentState[1] == 'East'):
            moveDict[currentState[0]] = ((currentState[0][0] - 1, currentState[0][1]), 'East')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] - 1, currentState[0][1])))

        if (currentState[1] == 'West'):
            moveDict[currentState[0]] = ((currentState[0][0] + 1, currentState[0][1]), 'West')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] + 1, currentState[0][1])))
           """ 
        if problem.isGoalState(currentState[0]):
            #print("found goal at " + str(currentState[0]))
            #print("aaa" + str(moveDict[(5, 3)]))
            """finalList = []
            traceback = currentState[0]
            moveStack = util.Stack()
            count = 0
            while traceback != problem.getStartState() and count < 25:
                moveStack.push(moveDict[traceback][1])
                traceback = moveDict[traceback][0]
                #print(traceback)
                count = count + 1
            while not moveStack.isEmpty():
                print(finalList.append(moveStack.pop()))
                
            return finalList
            #return currentState[0]"""
            return currentState[1]
        if not currentState[0] in closed:
            closed.add(currentState[0])
            for triple in problem.getSuccessors(currentState[0]):
                currentPath = list(currentState[1])
                currentPath.append(triple[1])
                successorStack.push((triple[0], currentPath))
    print("left loop w.o. finding goal")
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actionList = []
    currentState = (problem.getStartState(), 0 , 0)
    closed = set([])
    #each state maps to its parent
    #moveDict = {}
    if problem.isGoalState(problem.getStartState()):
        return []
    successorStack = util.Queue()
    successorStack.push((problem.getStartState(), []))

    while not successorStack.isEmpty():
        currentState = successorStack.pop()
        """if (currentState[1] == 'North'):
            moveDict[currentState[0]] = ((currentState[0][0] , currentState[0][1] - 1), 'North')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] , currentState[0][1] - 1)))
        if (currentState[1] == 'South'):
            moveDict[currentState[0]] = ((currentState[0][0] , currentState[0][1] + 1), 'South')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] , currentState[0][1] + 1)))
        if (currentState[1] == 'East'):
            moveDict[currentState[0]] = ((currentState[0][0] - 1, currentState[0][1]), 'East')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] - 1, currentState[0][1])))

        if (currentState[1] == 'West'):
            moveDict[currentState[0]] = ((currentState[0][0] + 1, currentState[0][1]), 'West')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] + 1, currentState[0][1])))
           """ 
        if problem.isGoalState(currentState[0]):
            #print("found goal at " + str(currentState[0]))
            #print("aaa" + str(moveDict[(5, 3)]))
            """finalList = []
            traceback = currentState[0]
            moveStack = util.Stack()
            count = 0
            while traceback != problem.getStartState() and count < 25:
                moveStack.push(moveDict[traceback][1])
                traceback = moveDict[traceback][0]
                #print(traceback)
                count = count + 1
            while not moveStack.isEmpty():
                print(finalList.append(moveStack.pop()))
                
            return finalList
            #return currentState[0]"""
            return currentState[1]
        if not currentState[0] in closed:
            closed.add(currentState[0])
            for triple in problem.getSuccessors(currentState[0]):
                currentPath = list(currentState[1])
                currentPath.append(triple[1])
                successorStack.push((triple[0], currentPath))
    print("left loop w.o. finding goal")
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    actionList = []
    currentState = (problem.getStartState(), 0 , 0)
    closed = set([])
    #each state maps to its parent
    #moveDict = {}
    if problem.isGoalState(problem.getStartState()):
        return []
    successorStack = util.PriorityQueue()
    successorStack.push(((problem.getStartState(), []), 0), 0)

    while not successorStack.isEmpty():
        currentState = successorStack.pop()
        """if (currentState[1] == 'North'):
            moveDict[currentState[0]] = ((currentState[0][0] , currentState[0][1] - 1), 'North')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] , currentState[0][1] - 1)))
        if (currentState[1] == 'South'):
            moveDict[currentState[0]] = ((currentState[0][0] , currentState[0][1] + 1), 'South')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] , currentState[0][1] + 1)))
        if (currentState[1] == 'East'):
            moveDict[currentState[0]] = ((currentState[0][0] - 1, currentState[0][1]), 'East')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] - 1, currentState[0][1])))

        if (currentState[1] == 'West'):
            moveDict[currentState[0]] = ((currentState[0][0] + 1, currentState[0][1]), 'West')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] + 1, currentState[0][1])))
           """ 
        if problem.isGoalState(currentState[0][0]):
            #print("found goal at " + str(currentState[0][0]))
            #print("aaa" + str(moveDict[(5, 3)]))
            """finalList = []
            traceback = currentState[0]
            moveStack = util.Stack()
            count = 0
            while traceback != problem.getStartState() and count < 25:
                moveStack.push(moveDict[traceback][1])
                traceback = moveDict[traceback][0]
                #print(traceback)
                count = count + 1
            while not moveStack.isEmpty():
                print(finalList.append(moveStack.pop()))
                
            return finalList
            #return currentState[0]"""
            return currentState[0][1]
        if not currentState[0][0] in closed:
            closed.add(currentState[0][0])
            for triple in problem.getSuccessors(currentState[0][0]):
                currentPath = list(currentState[0][1])
                currentPath.append(triple[1])
                successorStack.push(((triple[0], currentPath), triple[2] + currentState[1]), triple[2] + currentState[1])
    print("left loop w.o. finding goal")
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    actionList = []
    currentState = (problem.getStartState(), 0 , 0)
    closed = set([])
    #each state maps to its parent
    #moveDict = {}
    if problem.isGoalState(problem.getStartState()):
        return []
    successorStack = util.PriorityQueue()
    successorStack.push(((problem.getStartState(), []), 0), 0)
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    while not successorStack.isEmpty():
        currentState = successorStack.pop()
        """if (currentState[1] == 'North'):
            moveDict[currentState[0]] = ((currentState[0][0] , currentState[0][1] - 1), 'North')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] , currentState[0][1] - 1)))
        if (currentState[1] == 'South'):
            moveDict[currentState[0]] = ((currentState[0][0] , currentState[0][1] + 1), 'South')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] , currentState[0][1] + 1)))
        if (currentState[1] == 'East'):
            moveDict[currentState[0]] = ((currentState[0][0] - 1, currentState[0][1]), 'East')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] - 1, currentState[0][1])))

        if (currentState[1] == 'West'):
            moveDict[currentState[0]] = ((currentState[0][0] + 1, currentState[0][1]), 'West')
            print("expanding" + str(currentState[0]) + "with its ancestor" + str((currentState[0][0] + 1, currentState[0][1])))
           """ 
        if problem.isGoalState(currentState[0][0]):
            #print("found goal at " + str(currentState[0][0]))
            #print("aaa" + str(moveDict[(5, 3)]))
            """finalList = []
            traceback = currentState[0]
            moveStack = util.Stack()
            count = 0
            while traceback != problem.getStartState() and count < 25:
                moveStack.push(moveDict[traceback][1])
                traceback = moveDict[traceback][0]
                #print(traceback)
                count = count + 1
            while not moveStack.isEmpty():
                print(finalList.append(moveStack.pop()))
                
            return finalList
            #return currentState[0]"""
            return currentState[0][1]
        if not currentState[0][0] in closed:
            closed.add(currentState[0][0])
            for triple in problem.getSuccessors(currentState[0][0]):
                currentPath = list(currentState[0][1])
                currentPath.append(triple[1])
                successorStack.push(((triple[0], currentPath), triple[2] + currentState[1]), triple[2] + currentState[1] + heuristic(triple[0], problem))
    print("left loop w.o. finding goal")
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
