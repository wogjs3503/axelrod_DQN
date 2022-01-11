import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
class DQN(Player):
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
        
        self.report_fifty = deque(maxlen=50)
        self.report_whole = []
        self.report_C = []
        self.report_D = []
        
        self.batch_size = 6
        
        self.terminal = False
        self.current_round=1
        
        try:
            f = open("number.txt", 'r')
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
        
        print ("hello")
        model = Sequential()
                
        model.add(Dense(16,input_dim=10 ,activation = 'relu'))
        model.add(Dense(16, activation = 'relu'))
        
        model.add(Dense(2, activation = 'relu'))
        
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
        
        self.action_selection_parameter = 0.8 - self.current_round / 3000 * 0.7
        if self.current_round < 0.1:
            self.action_selection_parameter = 0.1   
        
        if len(self.history) == 10:
            print (self.history)
        
        if len(self.history) == 0:
            self.prev_action = self.rnd_action()
            self.original_prev_action = self.prev_action
        
        state = self.find_state(opponent)
        reward = self.find_reward(opponent)
        
        #select action -> action을 적용한 reward, state 관찰 -> 데이터 저장 (batch에)
        #근데 이건, 마지막에 select action -> 다음 step 시작할 때 action 적용 -> 저장?..
        
        try:
            print (self.replay_memory[-1][1], self.replay_memory[-1][2])
        except:
            print ("sorry")
        
        """terminal 에 대해서 지정해줄 필요 있음. save, load 관련 o"""
        if self.current_round % 200==199:
            self.terminal = True
        else:
            self.terminal = False
        
        if (self.current_round % 200) > 11:
            self.update_replay_memory(self.prev_state, self.prev_action, reward, state, self.terminal)
            
        self.perform_training()
        
        action = self.select_action(state)
        if self.current_round >= 50:
            self.report_whole.append(self.report_fifty.count('C'))
        
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
                self.report_fifty.append('C')
                return C
            else:
                self.report_fifty.append('D')
                return D
            
        #입실론에 걸렸을 때
        if len(self.prev_state) >= 10:    
            index = np.argmax(self.get_q_values(self.prev_state)[0])
            
            if index == 0:
                self.report_fifty.append('C')
                return self.rnd_action()
            else:
                self.report_fifty.append('D')
                return self.rnd_action()
        
        self.report_fifty.append('X')
        try:
            self.report_C.append(self.report_C[-1])
            self.report_D.append(self.report_D[-1])
        except:
            print('hihi')
        return self.rnd_action()

    def get_q_values(self, x):
        
        temp=[]
        
        for i in range(len(x)):
            if (x[i]) == "C":
                temp.append(0)
            else:
                temp.append(1)  
        
        input_x = np.reshape(temp,[1,10])
        Q_value = self.model.predict(input_x)
        print(Q_value)
        self.report_C.append(Q_value[0][0])
        self.report_D.append(Q_value[0][1])
        return Q_value
    
    def find_state(self, opponent: Player) -> str:
        """
        Finds the my_state (the opponents last n moves +
        its previous proportion of playing C) as a hashable state
        """
        #여기서 아예 0101로 불러서 바꿔주자!!
        action_str = actions_to_str(opponent.history[-self.memory_length :])
        if len(opponent.history) == 10:
            print(opponent.history)
        return action_str

    def perform_training(self):

        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        samples = random.sample(self.replay_memory, self.batch_size)
        #samples = [replay_memory, replay_memory, replay_memory ...]
        #replay_memory = {prev_state, prev_action, reward, state, terminal}
        for j, (prev_state, action, reward, state, terminal) in enumerate(samples):
            #일단 전처리
            if action == C:
                action = 0
            elif action == D:
                action = 1

            #prev_state를 넣기좋은 형태로
            temp1 = []
            for i in range(len(prev_state)):
                if prev_state[i] == "C":
                    temp1.append(0)
                else:
                    temp1.append(1)
            input_prev_state = np.reshape(temp1, [1,10])
            
            #state를 넣기좋은 형태로
            temp2 = []
            for i in range(len(state)):
                if state[i] == "C":
                    temp2.append(0)
                else:
                    temp2.append(1)
            input_state = np.reshape(temp2, [1,10])
            
            target = reward
            if not self.terminal:
                target = reward + self.discount_rate * np.amax(self.model.predict(input_state)[0])
            
            target_f = self.model.predict(input_prev_state)
            target_f[j][action] = target
            
        self.model.fit(input_prev_state, target_f, batch_size=self.batch_size, verbose=0, shuffle = False)
        
        
        #current_input = [sample[0] for sample in samples]
        #current_input = [prev_state, prev_state, prev_state ...]
        #for j in range(len(current_input)): 
            #z=[]
            #for i in range(len(current_input[j])):
             #  if (current_input[j][i]) == "C":
             #      z.append(0)
             #  else:
             #       z.append(1)
            #y = np.reshape(z,(1,10,1))
            #prev_state를 input형태로 만들어줌
            #current_q_values = self.model.predict(y)
            #current_q_values[0]
            #print (current_q_values)
            
#            for i, (self.prev_state, state, action, reward) in enumerate(samples):
 #               #terminal 일 경우에 reward를 지급
  #              if terminal:
   #                 next_q_values = reward
    #            else:
     #              next_q_values = reward + self.discount_rate * np.max(next_q_values[i])
      #         print (current_q_values)
       #         print (next_q_values)
        #        current_q_values[i, action] = next_q_values
            
            #self.model.fit(y, current_q_values, batch_size=self.batch_size, verbose=0, shuffle=False)
            
        return                        

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
        return self.payoff_matrix[self.prev_action][opp_prev_action] - 2.75
    
    def _save(self, model_filepath):
        self.model.save(model_filepath)

    def save(self, suffix):
        self._save('model_{}.h5'.format(suffix))
        dic = {
        'replay_memory' : self.replay_memory,
        'score' : self.score,
        'current_round' : self.current_round,
        'report_fifty' : self.report_fifty,
        'report_whole' : self.report_whole,
        'report_C' : self.report_C,
        'report_D' : self.report_D
        }
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
        self.report_fifty = dic['report_fifty']
        self.report_whole = dic['report_whole']
        self.report_C = dic['report_C']
        self.report_D = dic['report_D']        