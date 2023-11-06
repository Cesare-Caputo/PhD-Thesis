# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:30:43 2020

@author: cesa_
"""

from garage_DP_class import Garage, cc_start
import numpy as np
from garage_cost import Exp_cost
from garage_demand import demand_static
import pandas as pd
state_space = [0,200,400,600,800,1000,1200,1400,1600,1800]
action_space= [0,1,2,3]

env = Garage()

def policy_evaluation(env, policy, gamma=.88, theta=1e-1,max_iterations=1e3):
    V_old = np.zeros(env.nS)
    while True:
        #new value function
        V_new = np.zeros(env.nS)
        #stopping condition
        delta = 0
        for s in state_space:
            Vs = 0
            action_possible = env.get_valid_action(s)
            for a in action_possible:
                 p, next_state, reward, done = env.step(s, a)
                 Vs += p * (reward + gamma * V_old[next_state//200])
            delta = np.abs(Vs - V_old[s//200])
            #update state-value
            V_new[s//200] = Vs
            
                #the new value function
        V_old = V_new

        #if true value function
        if delta < theta:
            break
    return V_old

def q_from_v(env, V, s, gamma=1):
    q= np.zeros(env.nA)
    action_possible = env.get_valid_action(s)
    for a in action_possible:
        prob, next_state, reward, done = env.step(s,a)
        V_next_state = V[next_state//200]
        q[a] += prob * (reward + gamma * V_next_state)
    return q

def q_greedify_policy(env, V, pi, s, gamma):
    """
    Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``.
    """
    ### START CODE HERE ###
    G = np.zeros_like(env.A, dtype=float)
    action_possible = env.get_valid_action(s)
    for a in action_possible:
        prob, next_state, reward, done = env.step(s,a)
        V_next_state = V[next_state//200]
        G[a] += prob * (reward + gamma * V_next_state)
            
    greed_actions = np.argwhere(G == np.amax(G))
    for a in env.A:
        if a in greed_actions:
            pi[s//200, a] = 1 / len(greed_actions)
        else:
            pi[s//200, a] = 0

def improve_policy(env, V, pi, gamma):
    policy_stable = True
    
    for s in state_space:
        old = pi[s//200].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s//200], old):
            policy_stable = False
    return pi, policy_stable

def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    while not policy_stable:
        V = policy_evaluation(env, pi)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
    return pi,V


V1 = np.zeros(env.nS)
def one_step_lookahead_1(env, state, V, discount_factor): # just for debugging
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        action_possible = env.get_valid_action(state*200)
        for a in action_possible:
            prob, next_state, reward, done = env.step(state*200,a)
            A[a] += prob * (reward + discount_factor * V[next_state//200])
        return A

def one_step_lookahead_2(env, state, V, discount_factor): # just for debugging, state is capacity
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        action_possible = env.get_valid_action(state)
        for a in action_possible:
            prob, next_state, reward, done = env.step(state,a)
            A[a] += prob * (reward + discount_factor * V[next_state//200])
        return A



def value_iteration(env, theta=1e-1, discount_factor=.88, max_iterations=1e3):
    """
    Value Iteration Algorithm.
    
    Args:
        env: 
            env.step returns ones step dynamics
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int) , as capacity value not integer
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        action_possible = env.get_valid_action(state)
        for a in action_possible:
            prob, next_state, reward, done = env.step(state,a)
            A[a] += prob * (reward + discount_factor * V[next_state//200])
        return A
    
    V = np.zeros(env.nS)
    # while True:
    #     # Stopping condition
    for i in range(int(max_iterations)):
        for s in state_space:
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = np.abs(best_action_value - V[s//200])
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s//200] = best_action_value
            #print(V[s])
            # Check if we can stop 
        if delta < theta:
                break
    print(f'Value-iteration converged at iteration#{i}.')
    print("Final Delta" , delta)
    # Create a deterministic policy using the optimal value function
    policy_det = np.zeros([env.nS, env.nA])
    for s in state_space:
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy_det[s//200, best_action] = 1.0
    
    return policy_det, V

policy_vi, v_vi = value_iteration(env)

# print("Policy Probability Distribution using value iteration:")
# print(policy)
# print("")


# # # delta debugging
# theta = 10300001.0
# V = np.zeros(env.nS)
# while True:
#     # Stopping condition
#     delta = 0
#     for state in state_space:
#         # Do a one-step lookahead to find the best action
#         A = one_step_lookahead_2(env,state, V, 1)
#         print(A)
#         best_action_value = np.max(A)
#         # Calculate delta across all states seen so far
#         delta = np.abs(best_action_value - V[state//200])
#         print("delta in state ", state)
#         print("is equal to ", delta)
#         # Update the value function. Ref: Sutton book eq. 4.10. 
#         V[state//200] = best_action_value
#         # print("Value function for state ", s) 
#         # print("is equal to", V[s//200])
#         # Check if we can stop 
#     if delta < theta:
#         # print(f'Value-iteration converged at iteration#{i}.')
#         break

# # Create a deterministic policy using the optimal value function
# policy_opt = np.zeros([env.nS, env.nA])
# for state in state_space:
#     # One step lookahead to find the best action for this state
#     A = one_step_lookahead_2(env,state, V,1)
#     best_action = np.argmax(A)
#     print("best action in state", state)
#     print("is to contruct floors" , best_action)
#     # Always take the best action
#     policy_opt[state//200, best_action] = 1.0


# Q = np.zeros([env.nS, env.nA])
# for s in state_space:
#     Q[s//200] = q_from_v(env, V, s)
# print("Action-Value Function:")
# print(Q)




#print the optimal policy
# print("\nOptimal Policy (Nothing = 0, 1 floor = 1, 2 floors = 2, 3 floors = 3):")
# print(policy_pi,"\n")

# action_pi =np.zeros(env.nS)
# for s in range(env.nS):
#     action_pi[s] = np.rint(np.argmax(policy_pi[s]))
    
def NPV_garage_DP(policy): # calculates the NPV when following a given determinstic policy
        NPV = 0
        years = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
        cc = 16000# Construction cost per parking space
        cl = 3600000# Annual leasing land cost
        #p = 0.00# Percentage of construction cost to acquire flexibility (i.e. fatter columns). Note that this is set to 0 to determine how much the flexibility is worth. The real value can be found by subracting the acquiistion cost from the NPV for a particular design.
        cr = 2000# Operating cost per parking space
        #ct = []# Total construction cost
        gc = 0.10# Growth in construction cost per floor above two floors
        n0 = 200# Initial number of parking space per floor
        p = 10000# Price per parking space
        r = 0.12# Discount rate
        fmin = 2# Minimum number of floors built
        fmax = 9# Maximum number of floors built
        T = 20
        kmax = n0*fmax
        k = pd.Series(index=years, name = 'capacity', dtype ='float64')
        revenue = pd.Series(index=years, dtype ='float64')
        CF = pd.Series(index=years,name ='Cash Flow', dtype ='float64')
        demand_projections = pd.Series(index=years, name = 'demand', dtype ='float64')
        opex = pd.Series(index=years, dtype ='float64')
        fixed_costs = pd.Series(index=years, dtype ='float64')
        E_cost = pd.Series(index=years, name = ' Construction costs', dtype ='float64')
        actions = pd.Series(index=years,name = 'action taken', dtype ='float64')
        E_cost[T]=0
        k[0] = 0
        #obtain actions for each state from policy
        pi =np.zeros(env.nS)
        for s in range(env.nS):
            pi[s] = np.rint(np.argmax(policy[s]))
        
        for i in range(0,21):
            if i ==20:
                k[i] = k[i-1]
                actions[i] = pi[int(k[i])//200]
                E_cost[i] = Exp_cost(k[i], actions[i] ) #remember to adjust this to year 0 AFTER BUG FIXED
            else:
                actions[i] = pi[int(k[i])//200]
                k[i+1] = k[i] + 200*actions[i] 
                E_cost[i] = Exp_cost(k[i], actions[i] )

            # if i ==0:
            #     actions[i] = pi[i]
            #     k[i+1] = actions[i]*200
            #     E_cost[i] = cc_start(actions[i]) # set starting cost equal to expansion here
            # else:
            #     actions[i] = pi[int(k[i])//200]                   
            #     k[i+1] = k[i] + 200*actions[i] 
            #     E_cost[i] = Exp_cost(k[i], actions[i] )
        E_cost[0] = cc_start(actions[0])
        for i in range(0,T+1):
            if i ==T:
                fixed_costs[i] = 0 #no leasing paid in last year
            else : 
                fixed_costs[i] = cl # leasing paid all years including 0CF
        for i in range(0,T+1): #initializing all ks to initial capacity but maybe change here to let k[0] = 0 
            demand_projections[i] = demand_static(i)
            revenue[i] = np.minimum(demand_projections[i], k[i])*p
            opex[i] = k[i] * cr
            CF[i] = revenue[i] - opex [i] - fixed_costs[i] - E_cost[i]
            NPV += CF[i]/((1+r)**i)
            model = pd.concat([CF, k, actions, demand_projections, E_cost], axis = 1)
        return  NPV, model, pi

# Create a deterministic policy from intuition for comparison
policy_opt = np.zeros([env.nS, env.nA])
policy_opt[0, 3] = 1.0
policy_opt[1, 3] = 1.0
policy_opt[2, 3] = 1.0
policy_opt[3, 3] = 1.0
policy_opt[4, 2] = 1.0
policy_opt[5, 3] = 1.0
policy_opt[6, 2] = 1.0
policy_opt[7, 1] = 1.0
policy_opt[8, 0] = 1.0
policy_opt[9, 0] = 1.0


#inputs
gamma = .88
init_policy =np.ones((10,4))/4
init_V = np.zeros(10)
theta = 100

#
V_pol = policy_evaluation(env, init_policy)
Q_pol, stable =  improve_policy(env, V_pol, init_policy, gamma)
Q_pol_iter, V_pol_iter = policy_iteration(env, gamma, theta)
# # NPV, model = NPV_garage_DP(policy_pi)
# NPV_Q, model_q, pi_q = NPV_garage_DP(Q_pol)
# NPV_Q_iter, model_q_iter, pi_q_iter = NPV_garage_DP(Q_pol_iter)
# NPV_init, model_init, pi_init = NPV_garage_DP(init_policy)
# NPV_v_iter_opt, model_v_iter_opt, pi_v_iter_opt = NPV_garage_DP(policy_opt)
# NPV_v_iter, model_v_iter, pi_v_iter = NPV_garage_DP(policy)
# print("NPV for optimal policy is", NPV)
# print("NPV for Q policy is", NPV_Q)
# print("NPV for policy iterated is", NPV_Q_iter)
# print("NPV for improved policy is", NPV_Q)
# print("NPV for value iteration policy is", NPV_v_iter)
# #error defnitely in policy iteration code
# what i thought was bug in NPV was correct whole time...#