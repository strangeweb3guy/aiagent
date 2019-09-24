import random
import pandas as pd
import numpy as np


environment_matrix = [[None, 0], # left out -> 1
                  [-100, 0], # 1<-2 = -100    1->2 = 0
                  [0, 0], # 2-3
                  [0, 0], # 4-5
                  [0, 100], # 5-6
                  [0, 0], # ----------- good
                  [100, 0], # 7-8
                  [0, 0], # 8-9
                  [0, 0], # 9-10
                  [0,-50], # 10-11
                  [0,None]] # 11- right out

'''
1
2
3
4
5
6
7
8
9
10
11
'''
q_matrix = [[0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
      [0,0]]

win_loss_states = [0,5,10]
def getAllPossibleNextAction(cur_pos):
    step_matrix = [x != None for x in environment_matrix[cur_pos]]
    action = []
    if(step_matrix[0]):
        action.append(0)
    if(step_matrix[1]):
        action.append(1)
    return(action)

def isGoalStateReached(cur_pos):
    return (cur_pos in [5])

def getNextState(cur_pos, action):
    if (action == 0):
        return cur_pos - 1
    else:
        return cur_pos + 1

def isGameOver(cur_pos):
    return cur_pos in win_loss_states

def test_action(cur_pos, posible_actions, lr):

    if random.random() > lr:
        if max(q_matrix[cur_pos])!=0:
            return q_matrix[cur_pos].index(max(q_matrix[cur_pos]))
            '''
            maxx = q_matrix[cur_pos].index(max(q_matrix[cur_pos]))
            minn = q_matrix[cur_pos].index(min(q_matrix[cur_pos]))
            if random.random() > 0.1:
                return maxx
            else:
                return minn
            '''
        elif posible_actions==[0]:
            return 0
        elif possible_actions==[1]:
            return 1
        else:
            return random.choice([0,1])
    else:
        return q_matrix[cur_pos].index(min(q_matrix[cur_pos]))


discount = 0.9
learning_rate = 0.1
for _ in range(1001):
    # get starting place
    cur_pos = random.choice(np.arange(len(environment_matrix)))
    print("Episode ", _ ,'\tpos:',cur_pos,'\n\t',cur_pos,"->" ,end='')

    # while goal state is not reached
    while(not isGameOver(cur_pos)):
        # get all possible next states from cur_step
        possible_actions = getAllPossibleNextAction(cur_pos)
        # select any one action randomly
        #action = random.choice(possible_actions)
        action = test_action(cur_pos, possible_actions, learning_rate)
        # find the next state corresponding to the action selected
        next_state = getNextState(cur_pos, action)
        # update the q_matrix
       # print('Q[{}][{}] += {}*({}+{}*{})\t(max({} -{}))'.format(cur_pos, action, learning_rate,environment_matrix[cur_pos][action],
        #                                       discount,max(q_matrix[next_state]) - q_matrix[cur_pos][action], q_matrix[next_state],
        #                                        q_matrix[cur_pos][action]))
        q_matrix[cur_pos][action]+= learning_rate * (environment_matrix[cur_pos][action] +
            discount * max(q_matrix[next_state]) - q_matrix[cur_pos][action])
        # go to next state
        cur_pos = next_state
        print(cur_pos, '->',end ='')
    # print status
    print( " done\n",'-'*20)
    #if _ == 10 or _== 100 or _==1000:
    print(pd.DataFrame(q_matrix))

print("Training done...")