import random 
import numpy as np
import torch 
from abc import ABC, abstractmethod, abstractproperty
from rlcard.agents.dqn_agent.typing import Transition

class Memory(ABC):
    ''' Abstract class for memory
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.max_memory_size = memory_size
        self.batch_size = batch_size

    @abstractproperty
    def memory(self) -> list: 
        pass 

    @abstractproperty
    def memory_size(self) -> int:
        pass 

    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if self.memory_size == self.max_memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def checkpoint_attributes(self):
        ''' Returns the attributes that need to be checkpointed
        '''
        
        return {
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'memory': self.memory
        }
            
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' 
        Restores the attributes from the checkpoint
        
        Args:
            checkpoint (dict): the checkpoint dictionary
            
        Returns:
            instance (Memory): the restored instance
        '''
        
        instance = cls(checkpoint['memory_size'], checkpoint['batch_size'])
        instance.memory = checkpoint['memory']
        return instance
    
    @abstractmethod
    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''

    @classmethod 
    def from_estimator_network(cls, estimator_network, memory_size, batch_size, max_sequence_length=None) -> "Memory":
        if estimator_network == 'mlp':
            return SimpleMemory(memory_size=memory_size, batch_size=batch_size)
        elif estimator_network == 'transformer':
            return SequenceMemory(memory_size=memory_size, batch_size=batch_size, max_sequence_length=max_sequence_length)
        raise ValueError(f"Estimator network {estimator_network} not supported")


class SimpleMemory(Memory):
    ''' Memory for saving transitions
    '''
    def __init__(self, memory_size, batch_size):
        super().__init__(memory_size=memory_size, batch_size=batch_size)
        self._memory = []

    @property
    def memory(self):
        return self._memory
    
    @memory.setter
    def memory(self, memory):
        self._memory = memory
    
    @property
    def memory_size(self):
        return len(self._memory)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        samples = tuple(zip(*samples))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)
    

# TODO (Kacper) Maybe move this to SequenceMemory class 
def post_pad_transitions(transitions: list[Transition], length: int) -> list[Transition]:
    """
    Post-pad the sequence of transitions with the last transition to make it a fixed length
    """
    return transitions + [Transition.padding_transition(transitions[-1]) for _ in range(length - len(transitions))]


class SequenceMemory(Memory):
    ''' Memory for saving sequences of transitions
    '''
    def __init__(self, memory_size, batch_size, max_sequence_length: int):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
            max_sequence_length (int): the maximum length of the sequence
        '''
        super().__init__(memory_size, batch_size)
        self.max_sequence_length = max_sequence_length
        self._memory = []

    @property
    def memory(self) -> list[Transition]:
        return self._memory
    
    @memory.setter
    def memory(self, memory):
        self._memory = memory

    @property 
    def memory_size(self):
        return len(self.memory)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of sequences of states
            action_batch (list): a batch of sequences of actions
            reward_batch (list): a batch of sequences of rewards
            next_state_batch (list): a batch of sequences of states
            done_batch (list): a batch of sequences of dones
        '''
        # TODO It is debatable whether padding the memory beyond self.max_sequence_length is a benefitial, as it might 
        # inflate the percentage of padded transitions in the traning distribution, which would bias the model towards 
        # the early game (one with few hands played thus far). 
        padded_memory = post_pad_transitions(self.memory, self.memory_size + self.max_sequence_length - 1)

        # Sample a batch of starting indices of sequences and get sequences up to max_sequence_length
        start_idx = torch.randint(0, self.memory_size, (self.batch_size, ))
        sequences = [padded_memory[i:i+self.max_sequence_length] for i in start_idx] # [batch_size, sequence_length, 5]

        # The processing below is a bit convoluted, but it's just mirroring what the SimpleMemory does.
        def unpack_and_cat_transitions(sequence: list[Transition]):
            last_not_padded_index = max(i for i, transition in enumerate(sequence) if transition.action != Transition.padding_value())
            transitions = (
                np.array([transition.state for transition in sequence]),
                sequence[last_not_padded_index].action, # Should be the last action in the sequence
                sequence[last_not_padded_index].reward, # TODO (Kacper) figure out if this should be the last reward or some combination of previous rewards
                np.array([transition.next_state for transition in sequence]), # This has to be a sequence 
                sequence[last_not_padded_index].done, # Sequence done if the last state done
                sequence[last_not_padded_index].legal_actions, # Take the last legal action
            )        
            return transitions 
        sequences = map(unpack_and_cat_transitions, sequences)
        samples = list(zip(*sequences))

        states = np.array(samples[0])
        actions = np.array(samples[1])
        rewards = np.array(samples[2])
        next_states = np.array(samples[3])
        dones = np.array(samples[4])
        legal_actions = samples[5]
        return states, actions, rewards, next_states, dones, legal_actions
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' 
        Restores the attributes from the checkpoint
        
        Args:
            checkpoint (dict): the checkpoint dictionary
            
        Returns:
            instance (Memory): the restored instance
        '''
        
        instance = cls(checkpoint['memory_size'], checkpoint['batch_size'], checkpoint['max_sequence_length'])
        instance.memory = checkpoint['memory']
        return instance
    
    def checkpoint_attributes(self):
        ''' Returns the attributes that need to be checkpointed
        '''
        
        return {
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'memory': self.memory, 
            'max_sequence_length': self.max_sequence_length
        }
