import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ns3gym import ns3env
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################################################################################################
# Other Functions ####################################################################################################################################
######################################################################################################################################################
#################################################################################
# CSV Function ##################################################################
#################################################################################
def csv2list(file_path):
    result = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            result.append(row)
    return result
#################################################################################
# MLB/MRO/Coordinator Functions #################################################
#################################################################################
def action_func_Mro(actions):
    env_actions = []

    Hom = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    Ttt = [100, 128, 256, 320, 480, 512, 640]
    
    Hom_len = len(Hom)
    for i in actions:
        env_action = divmod(i, Hom_len)
        Hom_action = Hom[env_action[0]]
        Ttt_action = Ttt[env_action[1]]

        env_actions.append(Ttt_action)
        env_actions.append(Hom_action)

    return env_actions

def action_func_Mlb(actions):
    env_actions = []

    CIO = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    for i in range(len(actions)):
        CIO_action = CIO[actions[i]]
        
        env_actions.append(CIO_action)
        
    return env_actions

def action_func_coordinator(actions):
    env_actions_coordinator = []
    actions = actions[0]

    for i in range(5) :
        env_actions_coordinator.append(actions % 4) 
        actions = int(actions/4)
    
    return env_actions_coordinator

def action_func_coordinator_cluster(actions, cluster_list):
    env_actions_coordinator = []

    for i in range(5):
        env_actions_coordinator.append(0)

    idx_cluster = 0
    for i in cluster_list:
        action = actions[idx_cluster]
        for j in i:
            action = actions[idx_cluster]
            env_actions_coordinator[j] = action % 4
            action = int(action/4)
        idx_cluster = idx_cluster + 1

    return env_actions_coordinator
#################################################################################
######################################################################################################################################################
def jains_index(allocations):

    if len(allocations) == 0:
        return 0 

    sum_allocations = sum(allocations)
    sum_squared_allocations = sum(x**2 for x in allocations)
    n = len(allocations)
    
    jain_index = (sum_allocations ** 2) / (n * sum_squared_allocations)
    return jain_index


######################################################################################################################################################
# Sum Tree Class #####################################################################################################################################
######################################################################################################################################################
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"
######################################################################################################################################################
    
######################################################################################################################################################
# Prioritize Experience Replay Class ################################################################################################################
######################################################################################################################################################
class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        # self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, dtype=torch.int64)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state = transition

        self.tree.add(self.max_priority, self.count)

        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)

            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -self.beta

        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(device),
            self.action[sample_idxs].to(device).unsqueeze(1),
            self.reward[sample_idxs].to(device).unsqueeze(1),
            self.next_state[sample_idxs].to(device),
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities.squeeze()):

            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
######################################################################################################################################################

######################################################################################################################################################
# DDQN Class #########################################################################################################################################
######################################################################################################################################################
class DDQNAgent:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = 20000

        self.memory = PrioritizedReplayBuffer(self.state_size, self.action_size, self.capacity)

        self.gamma = 0.95  
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.999
        self.steps_done = 0
        self.learning_rate = 0.001
        self.train_start = 500

        self.model = self._build_model()
        self.target_model = self._build_model()

        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)

        self.update_target_model()

        self.loss = 0

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,self.action_size)
        )
        return model.to(device)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):

        self.steps_done += 1
        
        act_values = self.model(torch.tensor(state, dtype=torch.float32, device=device))

        if np.random.rand() <= self.epsilon:
            print("Random Action")
            return torch.LongTensor([random.randrange(self.action_size)]).to(device) 
        else :
            print("Agent Action")
            return torch.argmax(act_values[0]).unsqueeze(0).to(device) 
    
    def remember(self, state, action, reward, next_state):

        self.memory.add((state, action, reward, next_state))
        
        if (self.epsilon > self.epsilon_end) :
            self.epsilon *= self.epsilon_decay
        print("Epsilon: ",self.epsilon)
    
    def learn(self, batch_size):
        
        if self.steps_done < self.train_start:
            print("Saved Data Num: ",self.steps_done)
            print("Not learning")
            return
        
        print("Saved Data Num: ",self.steps_done)
        print("Learning")

        batch, weights, tree_idxs = self.memory.sample(batch_size)

        weights = weights.to(device)

        print('tree indexes in learn function: ',tree_idxs)

        states, actions, rewards, next_states  = batch

        current_q = self.model(states).gather(1,actions)

        max_action = torch.argmax(self.model(next_states),dim=1).unsqueeze(0)
        max_actions = max_action.transpose(0,1)
        max_next_q = self.target_model(next_states).gather(1,max_actions)
        
        expected_q = rewards + (self.gamma * max_next_q)

        td_errors = torch.abs(current_q - expected_q)

        self.memory.update_priorities(tree_idxs, td_errors.cpu().detach().numpy())

        loss = (weights * F.mse_loss(current_q, expected_q)).mean()

        self.loss = loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
######################################################################################################################################################
        
######################################################################################################################################################
# Main function ######################################################################################################################################
######################################################################################################################################################       
EPISODES = 2000 
max_env_steps = 30 
port=1403
stepTime=0.5
startSim=0
seed=3
simArgs = {}
debug=True


if __name__ == "__main__" :

    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

    ObsSpace_Mro = 11
    ActSpace_Mro = 49

    ObsSpace_Mlb = 36
    ActSpace_Mlb = 9

    # Cluster
    cluster_list = [ [0], [1], [2,3,4] ]

    state_size = 30
    action_size = 4**5

    # Cluster 1 (BS 1)
    C1_state_size = 6
    C1_action_size = 4

    # Cluster 2 (BS 2)
    C2_state_size = 6
    C2_action_size = 4

    # Cluster 3 (BS 3, 4, 5)
    C3_state_size = 18
    C3_action_size = 4**3

    done = False
    batch_size = 64 

    break_ep = 0
    FullStep_Ep = []

    previous_CIO = [0] * 5
    previous_TTT = [0] * 5
    previous_HOM = [0] * 5
    actions_Mro_previous = [0] * 5
    actions_Mlb_previous = [0] * 5

    # Q-table for MRO
    Q1_Mro = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable1_QMRO_Mid.csv"))
    Q2_Mro = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable2_QMRO_Mid.csv"))
    Q3_Mro = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable3_QMRO_Mid.csv"))
    Q4_Mro = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable4_QMRO_Mid.csv"))
    Q5_Mro = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable5_QMRO_Mid.csv"))
    # Q=table for MLB
    Q1_Mlb = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable1_QMLB_Mid.csv"))
    Q2_Mlb = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable2_QMLB_Mid.csv"))
    Q3_Mlb = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable3_QMLB_Mid.csv"))
    Q4_Mlb = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable4_QMLB_Mid.csv"))
    Q5_Mlb = np.array(csv2list("/home/mnc/CADC/Qtable/mid/Qtable5_QMLB_Mid.csv"))

    # Agents of Clusters
    C1_Coordinator = DDQNAgent(C1_state_size, C1_action_size)

    C2_Coordinator = DDQNAgent(C2_state_size, C2_action_size)

    C3_Coordinator = DDQNAgent(C3_state_size, C3_action_size)

    torch.autograd.set_detect_anomaly(True)

    for ep in range(EPISODES):

        # Reset environment and get first new observation
        state = env.reset()
        # rAll_Mlb = 0
        # rAll_Mro = 0
        done = False

        print("ep(EPISODE) : ",ep)

        # QMRO
        #############################
        state_Mro = np.reshape(state['AverageVelocity'], [5,1])

        state1_Mro = int(state_Mro[0])
        state2_Mro = int(state_Mro[1])
        state3_Mro = int(state_Mro[2])
        state4_Mro = int(state_Mro[3])
        state5_Mro = int(state_Mro[4])

        # print("State_Mro: ",state_Mro)
        #############################

        # QLB
        #############################
        state_Mlb = np.reshape(state['enbMLBstate'], [5,1])
        state1_Mlb = int(state_Mlb[0])
        state2_Mlb = int(state_Mlb[1])
        state3_Mlb = int(state_Mlb[2])
        state4_Mlb = int(state_Mlb[3])
        state5_Mlb = int(state_Mlb[4])

        # print("State_Mlb: ",state_Mlb)
        #############################

        # The Q-Table learning algorithm
        for j in range(max_env_steps): 
            print("*******************************")
            print("Episode: %d"%(ep+1))
            print("Step: %d"%(j+1))

            # QMRO
            if j == 0: ############################################
                action1_Mro = np.argmax(Q1_Mro[state1_Mro, :].astype(float))
                action2_Mro = np.argmax(Q2_Mro[state2_Mro, :].astype(float))
                action3_Mro = np.argmax(Q3_Mro[state3_Mro, :].astype(float))
                action4_Mro = np.argmax(Q4_Mro[state4_Mro, :].astype(float))
                action5_Mro = np.argmax(Q5_Mro[state5_Mro, :].astype(float))

                actions_Mro = []
                actions_Mro.append(action1_Mro)
                actions_Mro.append(action2_Mro)
                actions_Mro.append(action3_Mro)
                actions_Mro.append(action4_Mro)
                actions_Mro.append(action5_Mro)

                env_actions_Mro = action_func_Mro(actions_Mro)
                ############################################

                # QLB
                ############################################
                action1_Mlb = np.argmax(Q1_Mlb[state1_Mlb, :].astype(float))
                action2_Mlb = np.argmax(Q2_Mlb[state2_Mlb, :].astype(float))
                action3_Mlb = np.argmax(Q3_Mlb[state3_Mlb, :].astype(float))
                action4_Mlb = np.argmax(Q4_Mlb[state4_Mlb, :].astype(float))
                action5_Mlb = np.argmax(Q5_Mlb[state5_Mlb, :].astype(float))
                
                actions_Mlb = []
                actions_Mlb.append(action1_Mlb)
                actions_Mlb.append(action2_Mlb)
                actions_Mlb.append(action3_Mlb)
                actions_Mlb.append(action4_Mlb)
                actions_Mlb.append(action5_Mlb)

                env_actions_Mlb = action_func_Mlb(actions_Mlb)
                ############################################
                

                for k in range(5):
                    print("BS {} QMRO actions HOM: {}  TTT: {}".format(k+1, env_actions_Mro[2*k+1], env_actions_Mro[2*k]))
                    print("BS {} QLB actions CIO: {} ".format(k+1, env_actions_Mlb[k]))
                
                env_actions = []
                for n in env_actions_Mlb:
                    env_actions.append(n)

                for m in env_actions_Mro:
                    env_actions.append(m)
            

                CurrentCio = np.array(env_actions[:5]) #CIO
                CurrentHom = np.array(env_actions[5::2][:5]) #HOM
                CurrentTtt = np.array(env_actions[6::2][:5]) #TTT

                print("current cio: ", CurrentCio)
                print("current hom: ", CurrentHom)
                print("current ttt: ", CurrentTtt)

                PreviousCio = np.array(previous_CIO) # previous_CIO
                PreviousHom = np.array(previous_HOM) # previous_HOM        
                PreviousTtt = np.array(previous_TTT) # previous_TTT

                print("previous cio: ", PreviousCio)
                print("previous hom: ", PreviousHom)
                print("previous ttt: ", PreviousTtt)

                cioAction = []
                homAction = []
                tttAction = []

                for i in range(5):
                    if CurrentCio[i] > PreviousCio[i]:
                        cioAction.append(1.0)
                    elif CurrentCio[i] == PreviousCio[i]:
                        cioAction.append(0.0)
                    else:
                        cioAction.append(-1.0)
                    
                    if CurrentHom[i] > PreviousHom[i]:
                        homAction.append(1.0)
                    elif CurrentHom[i] == PreviousHom[i]:
                        homAction.append(0.0)
                    else:
                        homAction.append(-1.0)

                    if CurrentTtt[i] > PreviousTtt[i]:
                        tttAction.append(1.0)
                    elif CurrentTtt[i] == PreviousTtt[i]:
                        tttAction.append(0.0)
                    else:
                        tttAction.append(-1.0)
                
                C1_cioAction = []
                C2_cioAction = []
                C3_cioAction = []

                C1_homAction = []
                C2_homAction = []
                C3_homAction = []

                C1_tttAction = []
                C2_tttAction = []
                C3_tttAction = []

                C1_AvgCqi = []
                C2_AvgCqi = []
                C3_AvgCqi = []

                C1_dlPrbUsage = []
                C2_dlPrbUsage = []
                C3_dlPrbUsage = []

                C1_BestCell = []
                C2_BestCell = []
                C3_BestCell = []

                AvgCqi = state['AvgCqi']
                dlPrbusage = state['dlPrbusage']
                BestCell = state['enbBestCell']

                # Cluster 1 
                # Initial State
                ######################################################################################
                C1_cioAction.extend(cioAction[i] for i in cluster_list[0])
                C1_homAction.extend(homAction[i] for i in cluster_list[0])
                C1_tttAction.extend(tttAction[i] for i in cluster_list[0])
                C1_AvgCqi.extend(AvgCqi[i] for i in cluster_list[0])
                C1_dlPrbUsage.extend(dlPrbusage[i] for i in cluster_list[0])
                C1_BestCell.extend(BestCell[i] for i in cluster_list[0])  


                C1_state1_coordinator = np.array(C1_cioAction)
                C1_state1_coordinator = np.reshape(C1_state1_coordinator, (1, 1))
                C1_state1_coordinator = C1_state1_coordinator.astype(np.float64)

                C1_state2_coordinator = np.array(C1_homAction)
                C1_state2_coordinator = np.reshape(C1_state2_coordinator, (1, 1))
                C1_state2_coordinator = C1_state2_coordinator.astype(np.float64)

                C1_state3_coordinator = np.array(C1_tttAction)
                C1_state3_coordinator = np.reshape(C1_state3_coordinator, (1, 1))
                C1_state3_coordinator = C1_state3_coordinator.astype(np.float64)

                C1_state4_coordinator = np.array(C1_AvgCqi)
                C1_state4_coordinator = np.reshape(C1_state4_coordinator, (1, 1))
                C1_state4_coordinator = C1_state4_coordinator.astype(np.float64)

                C1_state5_coordinator = np.array(C1_dlPrbUsage)
                C1_state5_coordinator = np.reshape(C1_state5_coordinator, (1, 1))
                C1_state5_coordinator = C1_state5_coordinator.astype(np.float64)

                C1_state6_coordinator = np.array(C1_BestCell)
                C1_state6_coordinator = np.reshape(C1_state6_coordinator, (1, 1))
                C1_state6_coordinator = C1_state6_coordinator.astype(np.float64)

                C1_Coordinator_state = np.concatenate( (C1_state1_coordinator, 
                                                     C1_state2_coordinator, 
                                                     C1_state3_coordinator, 
                                                     C1_state4_coordinator,
                                                     C1_state5_coordinator,
                                                     C1_state6_coordinator) )
                C1_Coordinator_state = np.reshape(C1_Coordinator_state, [1,C1_state_size])
                C1_Coordinator_state_array = np.array(C1_Coordinator_state)

                C1_Coordinator_state = C1_Coordinator_state_array
                ######################################################################################

                # Cluster 2 
                # Initial State
                ######################################################################################
                C2_cioAction.extend(cioAction[i] for i in cluster_list[1])
                C2_homAction.extend(homAction[i] for i in cluster_list[1])
                C2_tttAction.extend(tttAction[i] for i in cluster_list[1])
                C2_AvgCqi.extend(AvgCqi[i] for i in cluster_list[1])
                C2_dlPrbUsage.extend(dlPrbusage[i] for i in cluster_list[1])
                C2_BestCell.extend(BestCell[i] for i in cluster_list[1])  

                C2_state1_coordinator = np.array(C2_cioAction)
                C2_state1_coordinator = np.reshape(C2_state1_coordinator, (1, 1))
                C2_state1_coordinator = C2_state1_coordinator.astype(np.float64)

                C2_state2_coordinator = np.array(C2_homAction)
                C2_state2_coordinator = np.reshape(C2_state2_coordinator, (1, 1))
                C2_state2_coordinator = C2_state2_coordinator.astype(np.float64)

                C2_state3_coordinator = np.array(C2_tttAction)
                C2_state3_coordinator = np.reshape(C2_state3_coordinator, (1, 1))
                C2_state3_coordinator = C2_state3_coordinator.astype(np.float64)

                C2_state4_coordinator = np.array(C2_AvgCqi)
                C2_state4_coordinator = np.reshape(C2_state4_coordinator, (1, 1))
                C2_state4_coordinator = C2_state4_coordinator.astype(np.float64)
                
                C2_state5_coordinator = np.array(C2_dlPrbUsage)
                C2_state5_coordinator = np.reshape(C2_state5_coordinator, (1, 1))
                C2_state5_coordinator = C2_state5_coordinator.astype(np.float64)
                
                C2_state6_coordinator = np.array(C2_BestCell)
                C2_state6_coordinator = np.reshape(C2_state6_coordinator, (1, 1))
                C2_state6_coordinator = C2_state6_coordinator.astype(np.float64)

                C2_Coordinator_state = np.concatenate( (C2_state1_coordinator, 
                                                     C2_state2_coordinator, 
                                                     C2_state3_coordinator, 
                                                     C2_state4_coordinator,
                                                     C2_state5_coordinator,
                                                     C2_state6_coordinator) )
                C2_Coordinator_state = np.reshape(C2_Coordinator_state, [1,C2_state_size])
                C2_Coordinator_state_array = np.array(C2_Coordinator_state)
                
                C2_Coordinator_state = C2_Coordinator_state_array
                ######################################################################################

                # Cluster 3 
                # Initial State
                ######################################################################################
                C3_cioAction.extend(cioAction[i] for i in cluster_list[2])
                C3_homAction.extend(homAction[i] for i in cluster_list[2])
                C3_tttAction.extend(tttAction[i] for i in cluster_list[2])
                C3_AvgCqi.extend(AvgCqi[i] for i in cluster_list[2])
                C3_dlPrbUsage.extend(dlPrbusage[i] for i in cluster_list[2])
                C3_BestCell.extend(BestCell[i] for i in cluster_list[2])  

                C3_state1_coordinator = np.array(C3_cioAction)
                C3_state1_coordinator = np.reshape(C3_state1_coordinator, (3, 1))
                C3_state1_coordinator = C3_state1_coordinator.astype(np.float64)

                C3_state2_coordinator = np.array(C3_homAction)
                C3_state2_coordinator = np.reshape(C3_state2_coordinator, (3, 1))
                C3_state2_coordinator = C3_state2_coordinator.astype(np.float64)

                C3_state3_coordinator = np.array(C3_tttAction)
                C3_state3_coordinator = np.reshape(C3_state3_coordinator, (3, 1))
                C3_state3_coordinator = C3_state3_coordinator.astype(np.float64)

                C3_state4_coordinator = np.array(C3_AvgCqi)
                C3_state4_coordinator = np.reshape(C3_state4_coordinator, (3, 1))
                C3_state4_coordinator = C3_state4_coordinator.astype(np.float64)
                
                C3_state5_coordinator = np.array(C3_dlPrbUsage)
                C3_state5_coordinator = np.reshape(C3_state5_coordinator, (3, 1))
                C3_state5_coordinator = C3_state5_coordinator.astype(np.float64)
                
                C3_state6_coordinator = np.array(C3_BestCell)
                C3_state6_coordinator = np.reshape(C3_state6_coordinator, (3, 1))
                C3_state6_coordinator = C3_state6_coordinator.astype(np.float64)

                C3_Coordinator_state = np.concatenate( (C3_state1_coordinator, 
                                                     C3_state2_coordinator, 
                                                     C3_state3_coordinator, 
                                                     C3_state4_coordinator,
                                                     C3_state5_coordinator,
                                                     C3_state6_coordinator) )
                C3_Coordinator_state = np.reshape(C3_Coordinator_state, [1,C3_state_size])
                C3_Coordinator_state_array = np.array(C3_Coordinator_state)
                
                C3_Coordinator_state = C3_Coordinator_state_array
                ######################################################################################

                prev_state = state

            # Action
            ##########################################################################################
            # Action of each cluster
            C1_action_coordinator = C1_Coordinator.act(C1_Coordinator_state)
            C2_action_coordinator = C2_Coordinator.act(C2_Coordinator_state)
            C3_action_coordinator = C3_Coordinator.act(C3_Coordinator_state)

            C1_actions_coordinator = C1_action_coordinator.unsqueeze(1).cpu().numpy()
            C2_actions_coordinator = C2_action_coordinator.unsqueeze(1).cpu().numpy()
            C3_actions_coordinator = C3_action_coordinator.unsqueeze(1).cpu().numpy()
            
            env_index = []
            env_index.append(C1_action_coordinator.item())
            env_index.append(C2_action_coordinator.item())
            env_index.append(C3_action_coordinator.item())
            
            env_actions_coordinator = action_func_coordinator_cluster(env_index, cluster_list)

            env_actions_chose = [0] * 15

            for i in range(5) :
                if (env_actions_coordinator[i] == 0) :
                    env_actions_chose[i] = env_actions[i]
                    env_actions_chose[2*i+5] = env_actions[2*i+5]
                    env_actions_chose[2*i+6] = env_actions[2*i+6]
                    
                elif (env_actions_coordinator[i] == 1) :
                    env_actions_chose[i] = env_actions[i]
                    env_actions_chose[2*i+5] = previous_HOM[i]
                    env_actions_chose[2*i+6] = previous_TTT[i]
                    
                elif (env_actions_coordinator[i] == 2) :
                    env_actions_chose[i] = previous_CIO[i]
                    env_actions_chose[2*i+5] = env_actions[2*i+5]
                    env_actions_chose[2*i+6] = env_actions[2*i+6]

                elif (env_actions_coordinator[i] == 3) :
                    env_actions_chose[i] = previous_CIO[i]
                    env_actions_chose[2*i+5] = previous_HOM[i]
                    env_actions_chose[2*i+6] = previous_TTT[i]


            for k in range(5) :
                print("After Coordination, BS {} actions HOM: {}  TTT: {}".format(k+1, env_actions_chose[2*k+6], env_actions_chose[2*k+5]))
                print("After Coordination, BS {} actions CIO: {} ".format(k+1, env_actions_chose[k]))
                
                if (env_actions_coordinator[k] == 0) :
                    print(" MRO : O MLB : O ")
                elif (env_actions_coordinator[k] == 1) :    
                    print(" MRO : X  MLB : O ")
                elif (env_actions_coordinator[k] == 2) :    
                    print(" MRO : O MLB : X  ")
                elif (env_actions_coordinator[k] == 3) :
                    print(" MRO : X MLB : X  ")
    
            
            # Get new state and reward from environment
            if(j>1) :
                prev_state = new_state
            new_state, reward, done, _ = env.step(env_actions_chose)

            if new_state is None:
                if j != 27 :
                    break_ep = break_ep +1
                else:
                    FullStep_Ep.append(ep+1)
                
                break
            
            print("break_ep: ",break_ep)
            print("Full Step Episode: ",FullStep_Ep)

            Results = new_state['Results']
            print("Total RLF: ",Results[0])
            print("Total PP: ", Results[1])

            # QMRO
            ######################################
            new_state_Mro = np.reshape(new_state['AverageVelocity'], [5,1])

            new_state1_Mro = int(new_state_Mro[0])
            new_state2_Mro = int(new_state_Mro[1])
            new_state3_Mro = int(new_state_Mro[2])
            new_state4_Mro = int(new_state_Mro[3])
            new_state5_Mro = int(new_state_Mro[4])

            # QMRO choose action
            action_Mro_coordinator = [0] * 5

            for i in range(5):
                if (env_actions_coordinator[i] == 0 or env_actions_coordinator[i] == 2):
                    action_Mro_coordinator[i] = actions_Mro[i]
                elif (env_actions_coordinator[i] == 1 or env_actions_coordinator[i] == 3):
                    action_Mro_coordinator[i] = actions_Mro_previous[i]         

            
            state1_Mro = new_state1_Mro
            state2_Mro = new_state2_Mro
            state3_Mro = new_state3_Mro
            state4_Mro = new_state4_Mro
            state5_Mro = new_state5_Mro
            ######################################

            # QLB
            ######################################
            new_state_Mlb = np.reshape(new_state['enbMLBstate'], [5,1])

            new_state1_Mlb = int(new_state_Mlb[0])
            new_state2_Mlb = int(new_state_Mlb[1])
            new_state3_Mlb = int(new_state_Mlb[2])
            new_state4_Mlb = int(new_state_Mlb[3])
            new_state5_Mlb = int(new_state_Mlb[4])

            action_Mlb_coordinator = [0] * 5
            for i in range(5):
                if (env_actions_coordinator[i] == 0 or env_actions_coordinator[i] == 1):
                    action_Mlb_coordinator[i] = actions_Mlb[i]   
                elif (env_actions_coordinator[i] == 2 or env_actions_coordinator[i] == 3):
                    action_Mlb_coordinator[i] = actions_Mlb_previous[i]           

            state1_Mlb = new_state1_Mlb
            state2_Mlb = new_state2_Mlb
            state3_Mlb = new_state3_Mlb
            state4_Mlb = new_state4_Mlb
            state5_Mlb = new_state5_Mlb
  
            new_action1_MRO = np.argmax(Q1_Mro[state1_Mro, :].astype(float))
            new_action2_MRO = np.argmax(Q1_Mro[state2_Mro, :].astype(float))
            new_action3_MRO = np.argmax(Q1_Mro[state3_Mro, :].astype(float))
            new_action4_MRO = np.argmax(Q1_Mro[state4_Mro, :].astype(float))
            new_action5_MRO = np.argmax(Q1_Mro[state5_Mro, :].astype(float))

            new_actions_Mro = []
            new_actions_Mro.append(new_action1_MRO)
            new_actions_Mro.append(new_action2_MRO)
            new_actions_Mro.append(new_action3_MRO)
            new_actions_Mro.append(new_action4_MRO)
            new_actions_Mro.append(new_action5_MRO)

            new_env_actions_Mro = action_func_Mro(new_actions_Mro)
            
            actions_Mro = new_actions_Mro
            env_actions_Mro = new_env_actions_Mro

            new_action1_Mlb = np.argmax(Q1_Mlb[state1_Mlb, :].astype(float))
            new_action2_Mlb = np.argmax(Q2_Mlb[state2_Mlb, :].astype(float))
            new_action3_Mlb = np.argmax(Q3_Mlb[state3_Mlb, :].astype(float))
            new_action4_Mlb = np.argmax(Q4_Mlb[state4_Mlb, :].astype(float))
            new_action5_Mlb = np.argmax(Q5_Mlb[state5_Mlb, :].astype(float))

            new_actions_Mlb = []
            new_actions_Mlb.append(new_action1_Mlb)
            new_actions_Mlb.append(new_action2_Mlb)
            new_actions_Mlb.append(new_action3_Mlb)
            new_actions_Mlb.append(new_action4_Mlb)
            new_actions_Mlb.append(new_action5_Mlb)

            new_env_actions_Mlb = action_func_Mlb(new_actions_Mlb)
            env_actions_Mlb = new_env_actions_Mlb

            actions_Mlb = new_actions_Mlb

            ###################### coordinator new state  #################
            CurrentCio = np.array(new_env_actions_Mlb[:5]) #CIO            
            CurrentHom = np.array(new_env_actions_Mro[0::2][:5]) #HOM
            CurrentTtt = np.array(new_env_actions_Mro[1::2][:5]) #TTT

            PreviousCio = np.array(env_actions[:5]) # previous_CIO
            PreviousHom = np.array(env_actions[5::2][:5]) # previous_HOM        
            PreviousTtt = np.array(env_actions[6::2][:5]) # previous_TTT

            cioAction = []
            homAction = []
            tttAction = []

            print("Current CIO: ",CurrentCio)
            print("Current HOM: ",CurrentHom)
            print("Current TTT: ",CurrentTtt)

            print("Previous CIO: ",PreviousCio)
            print("Previous HOM: ",PreviousHom)
            print("Previous TTT: ",PreviousTtt)

            for t in range(5):
                if CurrentCio[t] > PreviousCio[t]:
                    cioAction.append(1.0)
                elif CurrentCio[t] == PreviousCio[t]:
                    cioAction.append(0.0)
                else:
                    cioAction.append(-1.0)
                    
                if CurrentHom[t] > PreviousHom[t]:
                    homAction.append(1.0)
                elif CurrentHom[t] == PreviousHom[t]:
                    homAction.append(0.0)
                else:
                    homAction.append(-1.0)

                if CurrentTtt[t] > PreviousTtt[t]:
                    tttAction.append(1.0)
                elif CurrentTtt[t] == PreviousTtt[t]:
                    tttAction.append(0.0)
                else:
                    tttAction.append(-1.0)
                
            C1_cioAction = []
            C2_cioAction = []
            C3_cioAction = []

            C1_homAction = []
            C2_homAction = []
            C3_homAction = []

            C1_tttAction = []
            C2_tttAction = []
            C3_tttAction = []

            C1_AvgCqi = []
            C2_AvgCqi = []
            C3_AvgCqi = []

            C1_dlPrbUsage = []
            C2_dlPrbUsage = []
            C3_dlPrbUsage = []
            
            C1_BestCell = []
            C2_BestCell = []
            C3_BestCell = []

            AvgCqi = new_state['AvgCqi']
            dlPrbusage = new_state['dlPrbusage']
            BestCell = new_state['enbBestCell']
            
            # Cluster 1 
            # Initial State
            ######################################################################################
            C1_cioAction.extend(cioAction[i] for i in cluster_list[0])
            C1_homAction.extend(homAction[i] for i in cluster_list[0])
            C1_tttAction.extend(tttAction[i] for i in cluster_list[0])
            C1_AvgCqi.extend(AvgCqi[i] for i in cluster_list[0])
            C1_dlPrbUsage.extend(dlPrbusage[i] for i in cluster_list[0])
            C1_BestCell.extend(BestCell[i] for i in cluster_list[0]) 
  
            C1_new_state1_coordinator = np.array(C1_cioAction)
            C1_new_state1_coordinator = np.reshape(C1_new_state1_coordinator, (1, 1))
            C1_new_state1_coordinator = C1_new_state1_coordinator.astype(np.float64)
            
            C1_new_state2_coordinator = np.array(C1_homAction)
            C1_new_state2_coordinator = np.reshape(C1_new_state2_coordinator, (1, 1))
            C1_new_state2_coordinator = C1_new_state2_coordinator.astype(np.float64)
            
            C1_new_state3_coordinator = np.array(C1_tttAction)
            C1_new_state3_coordinator = np.reshape(C1_new_state3_coordinator, (1, 1))
            C1_new_state3_coordinator = C1_new_state3_coordinator.astype(np.float64)
            
            C1_new_state4_coordinator = np.array(C1_AvgCqi)
            C1_new_state4_coordinator = np.reshape(C1_new_state4_coordinator, (1, 1))
            C1_new_state4_coordinator = C1_new_state4_coordinator.astype(np.float64)
            
            C1_new_state5_coordinator = np.array(C1_dlPrbUsage)
            C1_new_state5_coordinator = np.reshape(C1_new_state5_coordinator, (1, 1))
            C1_new_state5_coordinator = C1_new_state5_coordinator.astype(np.float64)
            
            C1_new_state6_coordinator = np.array(C1_BestCell)
            C1_new_state6_coordinator = np.reshape(C1_new_state6_coordinator, (1, 1))
            C1_new_state6_coordinator = C1_new_state6_coordinator.astype(np.float64)

            C1_new_Coordinator_state = np.concatenate( (C1_new_state1_coordinator, 
                                                 C1_new_state2_coordinator, 
                                                 C1_new_state3_coordinator, 
                                                 C1_new_state4_coordinator,
                                                 C1_new_state5_coordinator,
                                                 C1_new_state6_coordinator) )
            C1_new_Coordinator_state = np.reshape(C1_new_Coordinator_state, [1,C1_state_size])
            C1_new_Coordinator_state_array = np.array(C1_Coordinator_state)

            C1_new_Coordinator_state = C1_new_Coordinator_state_array
            ######################################################################################
                
            # Cluster 2 
            # Initial State
            ######################################################################################
            C2_cioAction.extend(cioAction[i] for i in cluster_list[1])
            C2_homAction.extend(homAction[i] for i in cluster_list[1])
            C2_tttAction.extend(tttAction[i] for i in cluster_list[1])
            C2_AvgCqi.extend(AvgCqi[i] for i in cluster_list[1])
            C2_dlPrbUsage.extend(dlPrbusage[i] for i in cluster_list[1])
            C2_BestCell.extend(BestCell[i] for i in cluster_list[1])  

            C2_new_state1_coordinator = np.array(C2_cioAction)
            C2_new_state1_coordinator = np.reshape(C2_new_state1_coordinator, (1, 1))
            C2_new_state1_coordinator = C2_new_state1_coordinator.astype(np.float64)
            
            C2_new_state2_coordinator = np.array(C2_homAction)
            C2_new_state2_coordinator = np.reshape(C2_new_state2_coordinator, (1, 1))
            C2_new_state2_coordinator = C2_new_state2_coordinator.astype(np.float64)
            
            C2_new_state3_coordinator = np.array(C2_tttAction)
            C2_new_state3_coordinator = np.reshape(C2_new_state3_coordinator, (1, 1))
            C2_new_state3_coordinator = C2_new_state3_coordinator.astype(np.float64)
            
            C2_new_state4_coordinator = np.array(C2_AvgCqi)
            C2_new_state4_coordinator = np.reshape(C2_new_state4_coordinator, (1, 1))
            C2_new_state4_coordinator = C2_new_state4_coordinator.astype(np.float64)
            
            C2_new_state5_coordinator = np.array(C2_dlPrbUsage)
            C2_new_state5_coordinator = np.reshape(C2_new_state5_coordinator, (1, 1))
            C2_new_state5_coordinator = C2_new_state5_coordinator.astype(np.float64) 
            
            C2_new_state6_coordinator = np.array(C2_BestCell)
            C2_new_state6_coordinator = np.reshape(C2_new_state6_coordinator, (1, 1))
            C2_new_state6_coordinator = C2_new_state6_coordinator.astype(np.float64)

            C2_new_Coordinator_state = np.concatenate( (C2_new_state1_coordinator, 
                                                 C2_new_state2_coordinator, 
                                                 C2_new_state3_coordinator, 
                                                 C2_new_state4_coordinator,
                                                 C2_new_state5_coordinator,
                                                 C2_new_state6_coordinator) )
            C2_new_Coordinator_state = np.reshape(C2_new_Coordinator_state, [1,C2_state_size])
            C2_new_Coordinator_state_array = np.array(C2_new_Coordinator_state)
            
            C2_new_Coordinator_state = C2_new_Coordinator_state_array
            ######################################################################################

            # Cluster 3
            # Initial State
            ######################################################################################
            C3_cioAction.extend(cioAction[i] for i in cluster_list[2])
            C3_homAction.extend(homAction[i] for i in cluster_list[2])
            C3_tttAction.extend(tttAction[i] for i in cluster_list[2])
            C3_AvgCqi.extend(AvgCqi[i] for i in cluster_list[2])
            C3_dlPrbUsage.extend(dlPrbusage[i] for i in cluster_list[2])
            C3_BestCell.extend(BestCell[i] for i in cluster_list[2])  

            C3_new_state1_coordinator = np.array(C3_cioAction)
            C3_new_state1_coordinator = np.reshape(C3_new_state1_coordinator, (3, 1))
            C3_new_state1_coordinator = C3_new_state1_coordinator.astype(np.float64)
        
            C3_new_state2_coordinator = np.array(C3_homAction)
            C3_new_state2_coordinator = np.reshape(C3_new_state2_coordinator, (3, 1))
            C3_new_state2_coordinator = C3_new_state2_coordinator.astype(np.float64)
        
            C3_new_state3_coordinator = np.array(C3_tttAction)
            C3_new_state3_coordinator = np.reshape(C3_new_state3_coordinator, (3, 1))
            C3_new_state3_coordinator = C3_new_state3_coordinator.astype(np.float64)
        
            C3_new_state4_coordinator = np.array(C3_AvgCqi)
            C3_new_state4_coordinator = np.reshape(C3_new_state4_coordinator, (3, 1))
            C3_new_state4_coordinator = C3_new_state4_coordinator.astype(np.float64)
        
            C3_new_state5_coordinator = np.array(C3_dlPrbUsage)
            C3_new_state5_coordinator = np.reshape(C3_new_state5_coordinator, (3, 1))
            C3_new_state5_coordinator = C3_new_state5_coordinator.astype(np.float64) 
        
            C3_new_state6_coordinator = np.array(C3_BestCell)
            C3_new_state6_coordinator = np.reshape(C3_new_state6_coordinator, (3, 1))
            C3_new_state6_coordinator = C3_new_state6_coordinator.astype(np.float64)
            C3_new_Coordinator_state = np.concatenate( (C3_new_state1_coordinator, 
                                                 C3_new_state2_coordinator, 
                                                 C3_new_state3_coordinator, 
                                                 C3_new_state4_coordinator,
                                                 C3_new_state5_coordinator,
                                                 C3_new_state6_coordinator) )
            C3_new_Coordinator_state = np.reshape(C3_new_Coordinator_state, [1,C3_state_size])
            C3_new_Coordinator_state_array = np.array(C3_new_Coordinator_state)
            
            C3_new_Coordinator_state = C3_new_Coordinator_state_array
            ######################################################################################

            bsStepPrb = new_state['enbStepPrb']
            bsStepRlf = new_state['enbStepRlf']
            bsStepPp = new_state['enbStepPp']

            C1_bsStepRlf = []
            C2_bsStepRlf = []
            C3_bsStepRlf = []

            C1_bsStepPp = []
            C2_bsStepPp = []
            C3_bsStepPp = []

            # ###########################################
            C1_bsStepRlf.extend(bsStepRlf[i] for i in cluster_list[0])
            C2_bsStepRlf.extend(bsStepRlf[i] for i in cluster_list[1])
            C3_bsStepRlf.extend(bsStepRlf[i] for i in cluster_list[2])

            C1_bsStepPp.extend(bsStepPp[i] for i in cluster_list[0])
            C2_bsStepPp.extend(bsStepPp[i] for i in cluster_list[1])
            C3_bsStepPp.extend(bsStepPp[i] for i in cluster_list[2])
            # ###########################################

            mlbEft = jains_index(bsStepPrb)

            C1_hoap = (0.7 * sum(C1_bsStepRlf) + 0.3 * sum(C1_bsStepPp))
            C2_hoap = (0.7 * sum(C2_bsStepRlf) + 0.3 * sum(C2_bsStepPp)) 
            C3_hoap = (0.7 * sum(C3_bsStepRlf) + 0.3 * sum(C3_bsStepPp)) 
            
            bsStepUeNum = prev_state['bsStepUeNum']
            C1_bsStepUeNum = sum(bsStepUeNum[i] for i in cluster_list[0])
            C2_bsStepUeNum = sum(bsStepUeNum[i] for i in cluster_list[1])
            C3_bsStepUeNum = sum(bsStepUeNum[i] for i in cluster_list[2])
            
            hoap_min = 0
            C1_hoap_max = 0.7*C1_bsStepUeNum
            C2_hoap_max = 0.7*C2_bsStepUeNum
            C3_hoap_max = 0.7*C3_bsStepUeNum
            
            C1_hoap_normalized = (C1_hoap - hoap_min) / (C1_hoap_max - hoap_min) 
            C2_hoap_normalized = (C2_hoap - hoap_min) / (C2_hoap_max - hoap_min) 
            C3_hoap_normalized = (C3_hoap - hoap_min) / (C3_hoap_max - hoap_min) 
            
            reward_weight_MRO = 0.9 
            reward_weight_MLB = 0.1 

            C1_coordi_reward = (reward_weight_MLB * mlbEft - reward_weight_MRO * C1_hoap_normalized) * 10
            C2_coordi_reward = (reward_weight_MLB * mlbEft - reward_weight_MRO * C2_hoap_normalized) * 10
            C3_coordi_reward = (reward_weight_MLB * mlbEft - reward_weight_MRO * C3_hoap_normalized) * 10

            C1_coordi_reward = np.reshape(C1_coordi_reward, [1,1])
            C1_coordi_reward = C1_coordi_reward[0:1, 0:1]
            C1_coordi_reward = np.round(C1_coordi_reward, 5)
            C2_coordi_reward = np.reshape(C2_coordi_reward, [1,1])
            C2_coordi_reward = C2_coordi_reward[0:1, 0:1]
            C2_coordi_reward = np.round(C2_coordi_reward, 5)
            C3_coordi_reward = np.reshape(C3_coordi_reward, [1,1])
            C3_coordi_reward = C3_coordi_reward[0:1, 0:1]
            C3_coordi_reward = np.round(C3_coordi_reward, 5)

            C1_Coordinator.remember(C1_Coordinator_state, C1_actions_coordinator, C1_coordi_reward, C1_new_Coordinator_state)
            C2_Coordinator.remember(C2_Coordinator_state, C2_actions_coordinator, C2_coordi_reward, C2_new_Coordinator_state)
            C3_Coordinator.remember(C3_Coordinator_state, C3_actions_coordinator, C3_coordi_reward, C3_new_Coordinator_state)

            C1_Coordinator_state = C1_new_Coordinator_state
            C2_Coordinator_state = C2_new_Coordinator_state
            C3_Coordinator_state = C3_new_Coordinator_state

            C1_Coordinator.learn(batch_size)
            C2_Coordinator.learn(batch_size)
            C3_Coordinator.learn(batch_size)

            if((j%13) == 0) :
                print("Target network update")
                C1_Coordinator.update_target_model()
                C2_Coordinator.update_target_model()
                C3_Coordinator.update_target_model()

            env_actions = []
            for n in new_env_actions_Mlb:
                env_actions.append(n)

            for m in new_env_actions_Mro:
                env_actions.append(m)    
                
            for k in range(5):
                previous_CIO[k] = env_actions_chose[k]
                previous_HOM[k] = env_actions_chose[2*k+5]
                previous_TTT[k] = env_actions_chose[2*k+6]
                actions_Mro_previous[k] = actions_Mro[k]
                actions_Mlb_previous[k] = actions_Mlb[k]
######################################################################################################################################################
