# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        distanceList = []
        foodList = currentGameState.getFood().asList()
        pacmanPos = successorGameState.getPacmanPosition()

        # Stop pacman from getting stuck
        if action == 'Stop':
            return -float("inf")

        # If we are going to hit a ghost, don't go there
        for ghostState in newGhostStates:
            if ghostState.getPosition() == pacmanPos and ghostState.scaredTimer is 0:
                return -float("inf") 
        
        # return the min dist, negated
        for food in foodList:
            dist = ( (pacmanPos[0] - food[0]) ** 2 + (pacmanPos[1] - food[1]) ** 2 ) ** 0.5
            distanceList.append(-dist) 

        return max(distanceList)
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

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # min/max functions for the player, ghost agents to find their values on the tree
        def getMaxValue(state, depth):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            pacLegalActions = state.getLegalActions(0)
            v = -(float("inf"))
            for action in pacLegalActions:
                nextState = state.generateSuccessor(0, action)
                cV = getMinValue(nextState, depth, 1)
                v = max(v, cV)
            return v 
        def getMinValue(state, depth, index):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            ghostLegalActions = state.getLegalActions(index)
            v = float("inf")
            if index == state.getNumAgents() - 1:
                for action in ghostLegalActions:
                    nextState = state.generateSuccessor(index, action)
                    curV = getMaxValue(nextState, depth - 1)
                    v = min(v, curV)
            else:
                for action in ghostLegalActions:
                    nextState = state.generateSuccessor(index, action)
                    curV = getMinValue(nextState, depth, index + 1)
                    v = min(v, curV)
            return v
        # top level calls so we remember the action we need to take
        pacLegalActions = gameState.getLegalActions(0)
        maxVal = -(float("inf"))
        maxAct = Directions.STOP
        for action in pacLegalActions:
            nextState = gameState.generateSuccessor(0, action)
            curVal = getMinValue(nextState, self.depth, 1)
            curAct = action
            if curVal > maxVal:
                maxVal = curVal
                maxAct = curAct
        return maxAct
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # same as minimax with pruning added
        def getMaxValue(state, depth, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            pacLegalActions = state.getLegalActions(0)
            v = -(float("inf"))
            for action in pacLegalActions:
                nextState = state.generateSuccessor(0, action)
                cV = getMinValue(nextState, depth, 1, alpha, beta)
                v = max(v, cV)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v 
        def getMinValue(state, depth, index, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            ghostLegalActions = state.getLegalActions(index)
            v = float("inf")
            if index == state.getNumAgents() - 1:
                for action in ghostLegalActions:
                    nextState = state.generateSuccessor(index, action)
                    curV = getMaxValue(nextState, depth - 1, alpha, beta)
                    v = min(v, curV)
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            else:
                for action in ghostLegalActions:
                    nextState = state.generateSuccessor(index, action)
                    curV = getMinValue(nextState, depth, index + 1, alpha, beta)
                    v = min(v, curV)
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            return v
        pacLegalActions = gameState.getLegalActions(0)
        maxVal = -(float("inf"))
        maxAct = Directions.STOP
        alpha = -(float("inf"))
        beta = float("inf")
        for action in pacLegalActions:
            nextState = gameState.generateSuccessor(0, action)
            curVal = getMinValue(nextState, self.depth, 1, alpha, beta)
            curAct = action
            if curVal > maxVal:
                maxVal = curVal
                maxAct = curAct
            if curVal > beta:
                return curVal
            alpha = max(alpha, curVal)
        return maxAct
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
        # basically minimax with expected values added together instead of acutal terminal values
        def getMaxValue(state, depth):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            pacLegalActions = state.getLegalActions(0)
            v = -(float("inf"))
            for action in pacLegalActions:
                nextState = state.generateSuccessor(0, action)
                cV = getMinValue(nextState, depth, 1)
                v = max(v, cV)
            return v 
        def getMinValue(state, depth, index):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            ghostLegalActions = state.getLegalActions(index)
            v = float(0)
            if index == state.getNumAgents() - 1:
                for action in ghostLegalActions:
                    nextState = state.generateSuccessor(index, action)
                    v += getMaxValue(nextState, depth - 1) * (1.0 / len(ghostLegalActions))
                    
            else:
                for action in ghostLegalActions:
                    nextState = state.generateSuccessor(index, action)
                    v += getMinValue(nextState, depth, index + 1) * (1.0 / len(ghostLegalActions))
            return v
        # top level to remember action to perform
        pacLegalActions = gameState.getLegalActions(0)
        maxVal = -(float("inf"))
        maxAct = Directions.STOP
        for action in pacLegalActions:
            nextState = gameState.generateSuccessor(0, action)
            curVal = getMinValue(nextState, self.depth, 1)
            curAct = action
            if curVal > maxVal:
                maxVal = curVal
                maxAct = curAct
      
        return maxAct

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # we want to win
    if currentGameState.isWin():
        return float("inf")
    # don't want to lose
    if currentGameState.isLose():
        return - float("inf")

    # variables to keep track of things
    score = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    foodPos = newFood.asList()
    
    # get the min food, we want to eat it
    minFood = float("inf")
    for food in foodPos:
        dist = util.manhattanDistance(food, currentGameState.getPacmanPosition())
        if dist < minFood:
            minFood = dist
    
    # get the closest ghost, we want to avioudu it
    disttoghost = float("inf")
    for i in range(1, currentGameState.getNumAgents()):
        nextdist = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(i))
        disttoghost = min(disttoghost, nextdist)

    # had to scale these to make it work out
    # farther away ghost is, the better
    score += max(disttoghost, 6.0) * .75
    # the closer we are to the closest food, the better
    score -= minFood * 1.0
    capsulelocations = currentGameState.getCapsules()
    # the less capsules/food on the board, the better
    score -= len(foodPos) * 1.0
    score -= len(capsulelocations) * 1.0
    return score 
# Abbreviation
better = betterEvaluationFunction

