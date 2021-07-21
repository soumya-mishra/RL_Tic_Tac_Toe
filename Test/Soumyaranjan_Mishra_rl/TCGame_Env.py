from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""

        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        state = np.reshape(curr_state, (3, 3))
        max_val = 15.0  #objective is to make 15 points in a row, column or a diagonal
        
        #np.nansum() treats "Not a Number" (NaN) to zero
        along_axis = (max_val in np.concatenate((np.nansum(state, axis=1), np.nansum(state, axis=0))))
        
        #np.trace() :Return the sum along diagonals of the array.
        #np.fliplr() :Reverse the order of elements along axis 1
        
        return ((np.trace(state) == max_val) or (np.trace(np.fliplr(state)) == max_val) or (along_axis))
    

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)


    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state

    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal.
        Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""

        
        reward = -1 # -1 for each move agent takes (Initialize reward)
        
        #1  Transition of Agent from Current state to new state with an action 
        next_state = self.state_transition(curr_state, curr_action)
        terminal, result = self.is_terminal(next_state) # Check if next state is terminal  
        #Check rewards,,next_state and the step is terminal or not 
        if (result == 'Win'):   # Winning step gets 1o points 
            return next_state, 10, terminal   # +10 if the agent wins 
        elif (result == 'Tie'):     # Tie state get Zero points 
            return next_state, 0, terminal   # 0 if the game ends in a draw

        #2 state  determinations after actions
        
        agent_actions, env_actions = self.action_space(next_state)
        env_random_action = random.choice(list(env_actions))
        next_state = self.state_transition(next_state, env_random_action)

        terminal, result = self.is_terminal(next_state)

       #Check rewards,,next_state and the step is terminal or not 
        if (result == 'Win'):
            reward = -10    #-10 if the environment wins
        elif (result == 'Tie'):
            reward = 0
        return next_state, reward, terminal

    def reset(self):
        return self.state
