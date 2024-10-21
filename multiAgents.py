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
from pacman import GameState

from typing import NamedTuple, Union, Any, Sequence

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        '''My Code'''
        # Get dist to nearest ghost
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        ghostDist = nearestDist(newPos, ghostPositions)

        # Checking if ghosts are scared for the next 3 seconds
        safe = all(scaredTime >= 3 for scaredTime in newScaredTimes)

        #if ghosts are not scared and next move is ghost hit or brings pacman within one of a ghost, give max ghost score
        if not safe and ghostDist <= 1:
                ghostScore = 9999
        #otherwise ignore ghosts
        else:
            ghostScore = 0

        #calculate distance to nearest food
        foodDist = nearestDist(newPos, newFood.asList())


        #if next next position will result in food, give better food score
        if currentGameState.hasFood(newPos[0], newPos[1]):
            foodScore = 1 + 1/foodDist
        else:
            foodScore = 1 / foodDist

        return foodScore - ghostScore

#calculate the distance to the nearest position in positionlist from a given position
def nearestDist( pos, posList):

    minDist = 99999
    for posX in posList:
            dist = manhattanDistance(pos, posX)
            if dist < minDist: minDist = dist
    return minDist





def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)

        #call minimax for agent 1 every successor state of agent 0
        scores = []
        for move in legalMoves:
            successor = gameState.generateSuccessor(0, move)
            score = self.minimax_value(successor, 0, 1)
            scores.append(score)


        # Choose the action that maximises score, if multiple best actions choose randomly
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalMoves[chosenIndex]



    def minimax_value(self, state : GameState , depf, agent ):

        #if terminal state or max depth reached, return eval function result(Utility(s))
        if self.terminalTest( depf, state, state.getLegalActions(agent)):
            return self.evaluationFunction(state)

        #if agent is pacman return max value, increment agent and leave depth as is
        if agent == 0:
            scores =[]
            #for all successor states of agent, call minimax algorithm for new agent, get all scores
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent,action )
                scores.append(self.minimax_value( sucessor , depf, agent+1))
            return max(scores)

        #if agent is last agent in agents[] reset increment depth and reset agent to zero(pacman) for next recursive call
        #agent = ghost, return min value
        elif agent == state.getNumAgents() - 1:
            scores = []
            #for all successor states of agent, call minimax algorithm for new agent, get all scores
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent, action)
                scores.append(self.minimax_value( sucessor, depf+1, 0))
            return min(scores)

        #else agent is ghost, return min value, increment agent and leave depth as is
        else:
            scores = []
            #for all successor states of agent, call minimax algorithm for new agent, get all scores
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent, action)

                scores.append(self.minimax_value(sucessor , depf, agent+1))
            #print(scores)
            return min(scores)

    #check if state is terminal; win or lose or no more available actions, or if depth limit reached
    def terminalTest(self, depth, state: GameState,actions):
        if state.isLose() or state.isWin() or depth >= self.depth or not actions:
            return True
        return False






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)

        #initilise alpha and beta as negative infinity and infinity respectively
        alpha = -float("inf")
        beta = float("inf")

        #agent zero is a maximiser, initialise value as negative infinity
        v = -float("inf")

        #for all succesoor states of agnet 0 call the alpha beta algorithm for agent 1
        for move in legalMoves:
            successor = gameState.generateSuccessor(0, move)
            score = self.alphaBeta_value(successor, 0, 1, alpha, beta)

            #get max score
            if score > v :
                v = score
                bestmove = move

            #if max score greater than beta, prune (return current move)
            if v > beta:
                bestmove = move
                break
            #if score greater than alpha, update ,alpha
            alpha = max(alpha, v)

        return bestmove




    def alphaBeta_value(self, state : GameState , depf, agent, alpha, beta ):

        #if terminal state or max depth reached, return eval function result(Utility(s)
        if self.terminalTest( depf, state, state.getLegalActions(agent)):
            return self.evaluationFunction(state)

        #if agent is pacman= maximisier, initialise v as negative infinity
        if agent == 0:
            v = -float("inf")

            # for all succesor states of agnet 0 call the alpha beta algorithm for agent 1
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent, action)
                #get max score
                v = max(v, self.alphaBeta_value(sucessor, depf, agent + 1, alpha, beta))
                #If max score is greater than beta, prune : break and return v as max score
                if v > beta:
                    return v
                # if max score greater than alpha, update alpha
                alpha = max(alpha, v)
            return v

        # if agent is last agent in agents[] reset increment depth and reset agent to zero(pacman) for next recursive call
        # agent = ghost = minimiser, initialise v as infinity
        elif agent == state.getNumAgents() - 1:
            v = float("inf")
            # for all succesor states of last agnet call the alpha beta algorithm for agent 0
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent, action)
                # get min score
                v = min(v, self.alphaBeta_value(sucessor, depf + 1, 0, alpha, beta))
                # If min score is less than alpha, prune : break and return v as min score
                if v < alpha:
                    return v
                # if min score less than beta, update beta
                beta = min(beta, v)

            return v

        # else agent = ghost = minimiser, initialise v as infinity
        else:
            v = float("inf")
            # for all succesor states of  agnet call the alpha beta algorithm for agent+1
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent, action)
                # get min score
                v = min(v, self.alphaBeta_value(sucessor, depf, agent + 1, alpha, beta))
                # If min score is less than alpha, prune : break and return v as min score
                if v < alpha:
                    return v
                # if min score less than beta, update beta
                beta = min(beta, v)

            return v

    # check if state is terminal; win or lose or no more available actions, or if depth limit reached
    def terminalTest(self, depth, state: GameState,actions):
        if state.isLose() or state.isWin() or depth >= self.depth or not actions:
            return True
        return False




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)

        # agent zero is a maximiser, initialise value as negative infinity
        v = -float("inf")
        # for all succesoor states of agent 0 call the expectimax algorithm for agent 1
        for move in legalMoves:
            successor = gameState.generateSuccessor(0, move)
            score = self.expectimax_value(successor, 0, 1)
            # get max score
            if score > v:
                v = score
                bestmove = move

        return bestmove

    def expectimax_value(self, state: GameState, depf, agent):

        # if terminal state or max depth reached, return eval function result(Utility(s)
        if self.terminalTest(depf, state, state.getLegalActions(agent)):
            return self.evaluationFunction(state)

        #if agent is pacman= maximisier, initialise v as negative infinity
        if agent == 0:
            v = -float("inf")
            # for all succesor states of agent 0 call the alpha beta algorithm for agent 1
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent, action)
                # get max score
                v = max(v, self.expectimax_value(sucessor, depf, agent + 1))
            return v

        # if agent is last agent in agents[] reset increment depth and reset agnet to zero(pacman) for next recursive call
        # agent = ghost = get expected value, initialise v as 0
        elif agent == state.getNumAgents() - 1:
            v = 0
            #probability of a choice is random and therefore equal
            prob = 1 / len(state.getLegalActions(agent))
            # for all succesor states of last agent call the expectimax algorithm for agent 0
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent, action)
                #get the expected value, average value across all states
                v = v + prob* self.expectimax_value(sucessor, depf + 1, 0)
            return v

        # else agent = ghost = get expected value, initialise v as 0
        else:
            v = 0
            # probability of a choice is random and therefore equal
            prob = 1 / len(state.getLegalActions(agent))
            # for all successor states of  agent call the expectimax algorithm for agent+1
            for action in state.getLegalActions(agent):
                sucessor = state.generateSuccessor(agent, action)
                # get the expected value, average value across all states
                v = v + prob*  self.expectimax_value(sucessor, depf, agent + 1)
            return v

    # check if state is terminal; win or lose or no more available actions, or if depth limit reached
    def terminalTest(self, depth, state: GameState, actions):
        if state.isLose() or state.isWin() or depth >= self.depth or not actions:
            return True
        return False

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Used similar logic to reflex agent eval function.
    return score is a weighted sum combinaion of:
    gamescore + foodscore - ghostscore


    gamescore = score of current state
    foodscore = 2 * reciprocal of manhattan distance to nearest food
        rewards states that bring pacman closer to
        note that gamescore metric will inherently reward pacman for actually eating the food
    ghostscore:
        if all ghosts will be scared for next 4 seconds, reward hunting ghost:
        note that gamescore metric will inherently reward pacman for actually eating scared ghosts if they are adjacent
        ghost score = - 9 * reciprocal of manhattan distance to nearest ghost

        if neaest ghost will be within a manhattan distance of two, discourage pacman from going
        note that gamescore metric will inherently prevent pacman from being eaten
        ghostscore = 1

        else ignore ghosts
        ghostsocre = 0

    Issues:
    since manhattan distance is used for food  distances pacman may get stuck in a situation where they must move
    away from the nearest food to get it, e.g avoiding a wall
    The eval function this relies on a ghost "scaring" pacman into moving again,
    eval function may fail on maze with no ghost
    Eval function suited to use case, however if more generic and powerful function required,
    maze distance could be computed over manhattan distance
    """
    "*** YOUR CODE HERE ***"
    #collect state info
    pacPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = [ghost.getPosition() for ghost in ghostStates]
    gameScore = currentGameState.getScore()
    food = currentGameState.getFood().asList()

    #get dist to nearest ghost
    ghostDist = nearestDist(pacPos, ghostPositions)

    #get dist to nearest food
    foodDist = nearestDist(pacPos, food)

    #calculate ghost score
    if all(scaredTime >= 4 for scaredTime in ScaredTimes):
        ghostScore = -9/ghostDist
    elif (ghostDist<=2):
        ghostScore = 1
    else:
        ghostScore = 0

    #calculate foodscore
    foodScore = 2 / foodDist

    return gameScore + foodScore - ghostScore


# Abbreviation
better = betterEvaluationFunction

#calculate the distance to the nearest position in positionlist from a given position
def nearestDist( pos, posList):

    minDist = 99999

    for posX in posList:
            dist = manhattanDistance(pos, posX)
            if dist < minDist:
                minDist = dist

    return minDist




