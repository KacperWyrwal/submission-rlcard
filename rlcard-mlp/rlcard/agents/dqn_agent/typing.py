from typing import NamedTuple


class Transition(NamedTuple):
    state: list[int] 
    action: int
    reward: float
    next_state: list[int]
    done: bool
    legal_actions: list[int]

    # TODO There are multiple choices for padding the sequence. One would be to designate a value for the 
    # state, action, reward, next_state, done, and legal_actions, which would indicate a padded transition. 
    # In this case the next_state would likely have to be provided as an argument.
    # Anothor option would be to replicate the first or the last transition.
    @classmethod 
    def padding_value(cls):
        return -111 # TODO: Make sure this value is not a valid state, action, reward, next_state, done, or legal_actions.
    
    @classmethod
    def padding_transition(cls, transition: 'Transition'):
        return cls(
            state=[cls.padding_value()] * len(transition.state),
            action=cls.padding_value(),
            reward=cls.padding_value(),
            next_state=[cls.padding_value()] * len(transition.next_state),
            done=cls.padding_value(),
            legal_actions=[cls.padding_value()] * len(transition.legal_actions),
        )
    
    def clone(self):
        return self.__class__(self.state.copy(), self.action, self.reward, self.next_state.copy(), self.done, self.legal_actions.copy())
    

