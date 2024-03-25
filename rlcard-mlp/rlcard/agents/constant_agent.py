import numpy as np


ACTION_TO_INDEX = {
    'call': 0,
    'raise': 1,
    'fold': 2,
    'check': 3,
}


class ConstantAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, num_actions, action: str):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
            constant_action (int): The index of the action to be chosen
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.action_index = ACTION_TO_INDEX[action]

    def step(self, state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        # Always play the constant action if it is legal
        if self.action_index in state['legal_actions']:
            return self.action_index
        else:
            return np.random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        info = {}
        if self.action_index in state['legal_actions']:
            probs = {state['raw_legal_actions'][i]: (i == self.action_index) for i in range(len(state['legal_actions']))}
        else:
            probs = {state['raw_legal_actions'][i]: 1/len(state['legal_actions']) for i in range(len(state['legal_actions']))}
        info['probs'] = probs
        return self.step(state), info
    

LEDUC_NUM_ACTIONS = 4


class RaisingAgent(ConstantAgent):
    def __init__(self):
        super().__init__(LEDUC_NUM_ACTIONS, 'raise')


class CallingAgent(ConstantAgent):
    def __init__(self):
        super().__init__(LEDUC_NUM_ACTIONS, 'call')


class FoldingAgent(ConstantAgent):
    def __init__(self):
        super().__init__(LEDUC_NUM_ACTIONS, 'fold')


class CheckingAgent(ConstantAgent):
    def __init__(self):
        super().__init__(LEDUC_NUM_ACTIONS, 'check')
        


class LeducHoldemRuleAgentCall(object):
    ''' Leduc Hold 'em Rule agent version 3 that calls (or checks if no previous bet was made)
        with probability p.
    '''
    def __init__(self, p=0.5):
        self.use_raw = True
        self.p = p

    # @staticmethod
    def step(self, state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''
        random_number = random.random()
        legal_actions = state['raw_legal_actions']

        # Aggressively play 'call' (i.e. never when there is the opportunity to call/check)
        # Else simply fold.

        if 'call' in legal_actions and random_number < self.p:
            return 'call'
        if 'check' in legal_actions and random_number < self.p:
            return 'check'
        else:
            return 'fold'

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []


class LeducHoldemRuleAgentBluff(object):
    ''' Leduc Hold 'em Rule agent version 4 that bluffs with probability p.
        Bluffing: could either raise(aggressive bluff) or call(passive bluff)
        * if no public card is shown, raise with probability p
        * if public card, raise with probability p if the public card is NOT the same RANK as the hand card

    '''
    def __init__(self, aggressive_p=0.5, passive_p=0.5):
        self.use_raw = True
        self.aggressive_p = aggressive_p
        self.passive_p = passive_p


    # @staticmethod
    def step(self, state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''
        # do we bluff aggressively or passively?
        aggressive_bluff = random.random() < self.aggressive_p
        passive_bluff = random.random() < self.passive_p

        legal_actions = state['raw_legal_actions']
        state = state['raw_obs']
        hand = state['hand']
        public_card = state['public_card']
        action = 'fold'

        # if no public card is shown, raise with probability aggressive_p or call with probability passive_p
        # (could add something to do with the strength of the rank: eg if J then bluff... if was K then not really bluffing)
        if not public_card:
            if aggressive_bluff and 'raise' in legal_actions:
                action = 'raise'
            # call or check depending on previous action
            elif passive_bluff and 'call' in legal_actions:
                action = 'call'
            elif passive_bluff and 'check' in legal_actions:
                action = 'check'
            else:
                action = 'fold'
        # if there is a public card:
        else:
            # if aggressive bluff and the public card is not the same rank as the hand card, raise
            if aggressive_bluff and 'raise' in legal_actions and public_card[1] != hand[1]:
                action = 'raise'
            # if passive bluff and the public card is not the same rank as the hand card, call
            elif passive_bluff and 'call' in legal_actions and public_card[1] != hand[1]:
                action = 'call'
            elif passive_bluff and 'check' in legal_actions and public_card[1] != hand[1]:
                action = 'check'
            else:
                action = 'fold'


        return action

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []
    


import random 


class LeducHoldemRuleAgentCall(object):
    ''' Leduc Hold 'em Rule agent version 3 that calls (or checks if no previous bet was made)
        with probability p.
    '''
    def __init__(self, p=0.5):
        self.use_raw = True
        self.p = p

    # @staticmethod
    def step(self, state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''
        random_number = random.random()
        legal_actions = state['raw_legal_actions']

        # Aggressively play 'call' (i.e. never when there is the opportunity to call/check)
        # Else simply fold.

        if 'call' in legal_actions and random_number < self.p:
            return 'call'
        if 'check' in legal_actions and random_number < self.p:
            return 'check'
        else:
            return 'fold'

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []


class LeducHoldemRuleAgentBluff(object):
    ''' Leduc Hold 'em Rule agent version 4 that bluffs with probability p.
        Bluffing: could either raise(aggressive bluff) or call(passive bluff)
        * if no public card is shown, raise with probability p
        * if public card, raise with probability p if the public card is NOT the same RANK as the hand card

    '''
    def __init__(self, aggressive_p=0.5, passive_p=0.5):
        self.use_raw = True
        self.aggressive_p = aggressive_p
        self.passive_p = passive_p


    # @staticmethod
    def step(self, state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''
        # do we bluff aggressively or passively?
        aggressive_bluff = random.random() < self.aggressive_p
        passive_bluff = random.random() < self.passive_p

        legal_actions = state['raw_legal_actions']
        state = state['raw_obs']
        hand = state['hand']
        public_card = state['public_card']
        action = 'fold'

        # if no public card is shown, raise with probability aggressive_p or call with probability passive_p
        # (could add something to do with the strength of the rank: eg if J then bluff... if was K then not really bluffing)
        if not public_card:
            if aggressive_bluff and 'raise' in legal_actions:
                action = 'raise'
            # call or check depending on previous action
            elif passive_bluff and 'call' in legal_actions:
                action = 'call'
            elif passive_bluff and 'check' in legal_actions:
                action = 'check'
            else:
                action = 'fold'
        # if there is a public card:
        else:
            # if aggressive bluff and the public card is not the same rank as the hand card, raise
            if aggressive_bluff and 'raise' in legal_actions and public_card[1] != hand[1]:
                action = 'raise'
            # if passive bluff and the public card is not the same rank as the hand card, call
            elif passive_bluff and 'call' in legal_actions and public_card[1] != hand[1]:
                action = 'call'
            elif passive_bluff and 'check' in legal_actions and public_card[1] != hand[1]:
                action = 'check'
            else:
                action = 'fold'


        return action

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []
