import random
import numpy as np
import pickle
import os

from collections import deque
from typing import Dict, Union

from axelrod.action import Action, actions_to_str
from axelrod.player import Player

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


Score = Union[int, float]

C, D = Action.C, Action.D


#에피소드를 총 10번 정도 돌리고, 각 경기당 match 개수를 30번 정도로?
class DQN_tester(Player):
    """A player who learns the best strategies through DQN algorithm.

    Names:

    - DQN trainer & agent
    """

    name = "DQN"
    classifier = {
        "memory_depth": float("inf"),  # Long memory
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }
    learning_rate = 0.01
    discount_rate = 0.98
    min_memory_length = 12
    memory_length = 10
    decay = 0.995
    
    def __init__(self) -> None:
        """Initialises the player by picking a random strategy."""

        super().__init__()

        # Set this explicitely, since the constructor of super will not pick it up
        # for any subclasses that do not override methods using random calls.
        
        self.classifier["stochastic"] = True

        self.prev_action = None  # type: Action
        self.original_prev_action = None  # type: Action
        
        self.score = 0

        self.prev_state = ""
        
        self.model = self.create_model()
        
        self.min_replay_memory_size = 30   
        self.replay_memory = deque(maxlen=30)
        self.batch_size = 6
        self.current_round = 1
        self.terminal = False

        try:
            f = open("testnumber.txt", 'r')
            suffix = int(f.readline())
            f.close()
            self.load(suffix)
        except OSError:
            print('I love hug!')

        self.action_selection_parameter = 0

        
    def receive_match_attributes(self):
        (R, P, S, T) = self.match_attributes["game"].RPST()
        self.payoff_matrix = {C: {C: R, D: S}, D: {C: T, D: P}}

    def create_model(self):
        
        model = Sequential()
                
        model.add(Dense(16,input_dim=10 ,activation = 'softmax'))
        model.add(Dense(16, activation = 'softmax'))
        
        model.add(Dense(2, activation = 'softmax'))
        
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        
        return model
    
    def rnd_action(self):
        p = random.random()
        if p>0.5:
            return C
        else:
            return D
    
    def strategy(self, opponent: Player) -> Action:
        """Runs a qlearn algorithm while the tournament is running."""        
        self.receive_match_attributes()
        
        if self.current_round == 60040:
            return
        if len(self.history) == 10:
            print (self.history)
        
        if len(self.history) == 0:
            self.prev_action = self.rnd_action()
            self.original_prev_action = self.prev_action
        
        state = self.find_state(opponent)
        reward = self.find_reward(opponent)
        
        try:
            print (self.replay_memory[-1][1], self.replay_memory[-1][2])
        except:
            print ("sorry")
        
        """terminal 에 대해서 지정해줄 필요 있음. save, load 관련 o"""
        #if self.current_round % 400==399:
        #    self.terminal = True
        #else:
        #    self.terminal = False
       
        self.terminal = False
        
        if (self.current_round % 400) > 11:
            self.update_replay_memory(self.prev_state, self.prev_action, reward, state, self.terminal)
                    
        action = self.select_action(state)
        #여기서 print "in prob"
        
        self.prev_state = state
        self.prev_action = action
        
        if self.terminal:
            self.save(self.current_round)

        print (self.current_round)        
        self.current_round += 1
        
        return action

    def update_replay_memory(self, prev_state, prev_action, reward, state, terminal):
        self.replay_memory.append((prev_state, prev_action, reward, state, terminal))
    
    def select_action(self, state: str) -> Action:
        """
        Selects the action based on the epsilon-soft policy
        """
        rnd_num = random.random()
        p = 1 - self.action_selection_parameter
                
        #정상적인 루트
        if rnd_num < p and len(self.prev_state) >= 10:
            index = np.argmax(self.get_q_values(self.prev_state)[0])

            if index == 0:
                return C
            else:
                #return D
                return T
            
        #입실론에 걸렸을 때
        return self.rnd_action()

    def get_q_values(self, x):
        
        temp=[]
        
        for i in range(len(x)):
            #if (x[i]) == "C":
            if random.random()<0.5:
                temp.append(0)
            else:
                temp.append(1)  
        print(temp)
        input_x = np.reshape(temp,[1,10])
        Q_value = self.model.predict(input_x)
        print(Q_value)
        return Q_value
    
    def find_state(self, opponent: Player) -> str:
        """
        Finds the my_state (the opponents last n moves +
        its previous proportion of playing C) as a hashable state
        """
        #여기서 아예 0101로 불러서 바꿔주자!!
        if len(opponent.history) == 10:
            print(opponent.history)
        action_str = actions_to_str(opponent.history[-self.memory_length :])
        return action_str                 

    def find_reward(self, opponent: Player) -> Dict[Action, Dict[Action, Score]]:

        #q-learning과 DQN 거의 동일할듯
        if len(opponent.history) == 0:
            opp_prev_action = self.rnd_action()
        else:
            opp_prev_action = opponent.history[-1]
        
        if self.prev_action == 'C':
            self.prev_action = C
        elif self.prev_action == 'D':
            self.prev_action = D
        
        return self.payoff_matrix[self.prev_action][opp_prev_action]
    
    def _save(self, model_filepath):
        self.model.save(model_filepath)

    def save(self, suffix):
        self._save('model_{}.h5'.format(suffix))
        dic = {
        'replay_memory' : self.replay_memory,
        'score' : self.score,
        'current_round' : self.current_round
        }
        print(dic)
        with open("training_info_{}.pkl".format(suffix), 'wb') as fout:
            pickle.dump(dic, fout)
        f = open("number.txt", 'w')
        f.write(str(self.current_round))
        f.close()

    def _load(self, model_filepath):
        self.model = keras.models.load_model(model_filepath)
        
    def load(self, suffix):
        self._load('model_{}.h5'.format(suffix))

        with open('training_info_{}.pkl'.format(suffix), 'rb') as fin:
            dic = pickle.load(fin)

        self.replay_memory = dic['replay_memory']
        self.score = dic['score']
        self.current_round = dic['current_round'] + 1