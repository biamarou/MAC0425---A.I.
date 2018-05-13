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
import math as th

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
        numOldFood = currentGameState.getFood()
        numOldFood = len(numOldFood.asList())

        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newFood = successorGameState.getFood()
        numNewFood = len(newFood.asList())
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if (successorGameState.isWin()):
            evaluation = float('inf')
        
        else:
            nearGhost = float('inf')
            nearFood = float('inf')
            if (newScaredTimes[0] == 0):           
                for g in newGhostStates:
                    p = g.getPosition()
                    ghostDistance = float(abs(newPos[0] - p[0]) + abs(newPos[1] - p[1]))
                    if (ghostDistance == 1 or ghostDistance == 0): nearGhost = -float('inf')
                    if (ghostDistance < nearGhost): nearGhost = ghostDistance
                        
            for f in newFood.asList():
                foodDistance = float(abs(newPos[0] - f[0]) + abs(newPos[1] - f[1]))
                if (foodDistance < nearFood): nearFood = foodDistance
                
            evaluation = 0.01*nearGhost + 2.*1.0/nearFood + 2.1*(numOldFood - numNewFood)
        return evaluation

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

class actionCost():

    def __init__(a, c):
        self.action = a
        self.cost = c

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def maxAction(self, gameState, depth, agent, call):
        
        v = -float('inf')
        nextActions = gameState.getLegalActions(agent)
        nextAction = None

        for a in nextActions:
            s = gameState.generateSuccessor(agent, a)
            c = self.costAction(s, depth, 1, call + 1)
            
            if (v < c):
                v = c
                nextAction = a
        
        if (call == 0): 
            return nextAction

        elif (nextAction == None):
            v = self.evaluationFunction(gameState)
        
        return v

    def minAction(self, gameState, depth, agent, call):

        v = float('inf')
        nextActions = gameState.getLegalActions(agent)
        nbAgent = gameState.getNumAgents()

        for a in nextActions:
            s = gameState.generateSuccessor(agent, a)
            
            if (agent == nbAgent - 1):
                c = self.costAction(s, depth - 1, 0, call + 1)
            else:
                c = self.costAction(s, depth, agent + 1, call + 1)
    
            if (v > c):
                v = c

        if (not nextActions):
            v = self.evaluationFunction(gameState)
        
        return v

    def costAction(self, gameState, depth, agent, call):
        
        if (depth == 0):
            return self.evaluationFunction(gameState)
        
        if (agent == 0):
            return self.maxAction(gameState, depth, agent, call)

        else:
            return self.minAction(gameState, depth, agent, call)

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

        return self.costAction(gameState, self.depth, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxAction(self, gameState, alfa, beta, depth, agent, call):
        
        v = -float('inf')
        nextActions = gameState.getLegalActions(agent)
        nextAction = None

        for a in nextActions:
            s = gameState.generateSuccessor(agent, a)
            c = self.costAction(s, alfa, beta, depth, 1, call + 1)
            
            if (v < c):
                v = c
                nextAction = a

            if (v > beta): return v
            alfa = max(alfa, v)
        
        if (call == 0): 
            return nextAction

        elif (nextAction == None):
            v = self.evaluationFunction(gameState)
        
        return v

    def minAction(self, gameState, alfa, beta, depth, agent, call):

        v = float('inf')
        nextActions = gameState.getLegalActions(agent)
        nbAgent = gameState.getNumAgents()

        for a in nextActions:
            s = gameState.generateSuccessor(agent, a)
            
            if (agent == nbAgent - 1):
                c = self.costAction(s, alfa, beta, depth - 1, 0, call + 1)
            else:
                c = self.costAction(s, alfa, beta, depth, agent + 1, call + 1)
    
            if (v > c):
                v = c
            
            if (v < alfa): return v
            beta = min(beta, v)

        if (not nextActions):
            v = self.evaluationFunction(gameState)
        
        return v

    def costAction(self, gameState, alfa, beta, depth, agent, call):
        
        if (depth == 0):
            return self.evaluationFunction(gameState)
        
        if (agent == 0):
            return self.maxAction(gameState, alfa, beta, depth, agent, call)

        else:
            return self.minAction(gameState, alfa, beta, depth, agent, call)


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.costAction(gameState, -float('inf'), float('inf'), self.depth, 0, 0)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxAction(self, gameState, depth, agent, call):
        
        v = -float('inf')
        nextActions = gameState.getLegalActions(agent)
        nextAction = None

        for a in nextActions:
            s = gameState.generateSuccessor(agent, a)
            c = self.costAction(s, depth, 1, call + 1)
            
            if (v < c):
                v = c
                nextAction = a
        
        if (call == 0): 
            return nextAction

        elif (nextAction == None):
            v = self.evaluationFunction(gameState)
        
        return v

    def expAction(self, gameState, depth, agent, call):

        v = 0
        nextActions = gameState.getLegalActions(agent)
        nbAgent = gameState.getNumAgents()

        for a in nextActions:
            s = gameState.generateSuccessor(agent, a)
            p = 1.0/float(len(nextActions))

            
            if (agent == nbAgent - 1):
                c = self.costAction(s, depth - 1, 0, call + 1)
            else:
                c = self.costAction(s, depth, agent + 1, call + 1)
    
            v += p * float(c)
        
        return v

    def costAction(self, gameState, depth, agent, call):
        
        if (depth == 0):
            return self.evaluationFunction(gameState)
        
        if (agent == 0):
            return self.maxAction(gameState, depth, agent, call)

        else:
            return self.expAction(gameState, depth, agent, call)


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.costAction(gameState, self.depth, 0, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    curPos = currentGameState.getPacmanPosition()
    curGhostStates = currentGameState.getGhostStates()
    curFood = currentGameState.getFood()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]

    if (currentGameState.isWin()):
            evaluation = float('inf')
        
    else:

        score = 0
                       
        for g in curGhostStates:
            p = g.getPosition()
            ghostDistance = float(abs(curPos[0] - p[0]) + abs(curPos[1] - p[1]))
                
            if (ghostDistance == 0): score = -float('inf')
            else: score += 4*ghostDistance

        score += -16.*len(curFood.asList())        
    
    return score + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction