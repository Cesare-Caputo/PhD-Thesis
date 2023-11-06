# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:30:43 2020

@author: cesa_
"""

from garage_DP_class import Garage, Garage_stoch,  cc_start, Garage_discounted, Garage_discounted_stoch
from backward_induction_dp import StochasticDP
import numpy as np
from garage_cost import Exp_cost
from garage_demand import demand_static,  demand_stochastic_less
import pandas as pd
state_space = [0,200,400,600,800,1000,1200,1400,1600,1800]
action_space= np.array([0,1,2,3])

env = Garage()
env_s = Garage_stoch()
env_d = Garage_discounted()
env_d_s = Garage_discounted_stoch()

def policy_evaluation(env, policy, discount_factor=.88, theta=1e-1,max_iterations=1e3):
    V = np.zeros(env.nS)
    for i in range(int(max_iterations)):
        delta = 0
        for s in state_space:
            v = 0
            for a, action_prob in enumerate(policy[s//200]):
                 p, next_state, reward, done = env.step(s, a)
                 v += action_prob*p * (reward + discount_factor * V[next_state//200])
            delta = max(delta,np.abs(V[s//200]-v))
            V[s//200] = v
        if delta < theta:
            break
    # print("Final Delta" , delta)
    return np.array(V)


def policy_improvement_coursera(env, discount_factor=.88):
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider ( , as capacity value not integer
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        action_possible = env.get_valid_action(state)
        for a in action_space:
            prob, next_state, reward, done = env.step(state,a)
            A[a] += prob * (reward + discount_factor * V[next_state//200])
        return A
    def q_greedify_policy(env, V, policy, s):
        """
        Mutate ``policy`` to be greedy with respect to the q-values induced by ``V``.
        """

        G = one_step_lookahead(s, V)
    
        greed_actions = np.argwhere(G == np.amax(G))
        for a in env.A:
            if a in greed_actions:
                policy[s//200, a] = 1 / len(greed_actions)
            else:
                policy[s//200, a] = 0
    
    #start with random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
      # Evaluate the current policy
      V = policy_evaluation(env, policy, discount_factor)
      
      # Will be set to false if we make any changes to the policy
      policy_stable = True
      
      # For each state...
      for s in state_space:
          old = policy[s//200].copy()
          q_greedify_policy(env, V, policy, s)
          if not np.array_equal(policy[s//200], old):
            policy_stable = False
         
      
      # If the policy is stable we've found an optimal policy. Return it
      if policy_stable:
          return policy, V

def policy_improvement(env,  discount_factor=.88, max_iterations=1e3):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI environment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """

    def one_step_lookahead(state, V):
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
        for a in action_space:
            prob, next_state, reward, done = env.step(state,a)
            A[a] += prob * (reward + discount_factor * V[next_state//200])
        return A
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    for i in range(int(max_iterations)):
        # Evaluate the current policy
        V = policy_evaluation( env, policy,  discount_factor)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s*200, V)
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V

def value_iteration(env,  discount_factor, theta=1e-1, max_iterations=1e4):
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
            state: The state to consider  , as capacity value not integer hence the weird indexing
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



    
def NPV_garage_DP(policy, demand = ''): # calculates the NPV when following a given determinstic policy
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
        
        #set parametrs for stochastic demand
        rD0s = np.random.random_sample() # Realised demand in year 0
        rD10s = np.random.random_sample() # Realised additional demand by year 10
        rDfs =np.random.random_sample()# Realised additional demand after year 10        
        
        
        
        #obtain actions for each state from policy
        pi =np.zeros(env.nS)
        for s in range(env.nS):
            pi[s] = np.rint(np.argmax(policy[s]))
        
        for i in range(1,21): #initializing all ks to initial capacity
            demand_projections[0] = 0
            if demand == 'stochastic':
                demand_projections[i] = demand_stochastic_less(i,rD0s,rD10s,rDfs)
            else:
                demand_projections[i] = demand_static(i)                
            
        for i in range(0,21):
            if i ==20:
                k[i] = k[i-1]
                actions[i] = pi[int(k[i])//200]
                E_cost[i] = Exp_cost(k[i], actions[i] ) #remember to adjust this to year 0 AFTER BUG FIXED
            else:
                actions[i] = pi[int(k[i])//200]
                k[i+1] = k[i] + 200*actions[i] 
                E_cost[i] = Exp_cost(k[i], actions[i] )

        E_cost[0] = cc_start(actions[0])
        for i in range(0,T+1):
            if i ==T:
                fixed_costs[i] = 0 #no leasing paid in last year
            else : 
                fixed_costs[i] = cl # leasing paid all years including 0CF
        for i in range(0,T+1): #initializing all ks to initial capacity but maybe change here to let k[0] = 0 
            revenue[i] = np.minimum(demand_projections[i], k[i])*p
            opex[i] = k[i] * cr
            CF[i] = revenue[i] - opex [i] - fixed_costs[i] - E_cost[i]
            NPV += CF[i]/((1+r)**i)
            model = pd.concat([CF, k, actions, demand_projections, E_cost], axis = 1)
        return  NPV, model



def ENPV_MC(nsim, policy):
    ENPV_res =[]
    for i in range(nsim):
        ENPV2, model2 = NPV_garage_DP(policy, demand = 'stochastic')
        ENPV_res.append(ENPV2)
    ENPV = np.mean(ENPV_res)   
    return ENPV



#inputs
gamma = .88
init_policy =np.ones((10,4))/4
init_V = np.zeros(10)
theta = 100



V_pol = policy_evaluation(env_s, init_policy)
#find optimal polic using policy iteration
policy_pi, v_pi = policy_improvement(env_s)
print("\nOptimal Policy with policy iteration (Nothing = 0, 1 floor = 1, 2 floors = 2, 3 floors = 3):")
print(policy_pi,"\n")
print("")
NPV_pi, model_pi = NPV_garage_DP(policy_pi)
print("NPV for policy iteration policy", NPV_pi)

#find optimal policy using value iteration with stochastic demand
policy_vi, v_vi = value_iteration(env, discount_factor = 1)
policy_vi_d, v_vi_d = value_iteration(env, discount_factor = .88)
policy_vi_s, v_vi_s = value_iteration(env_s, discount_factor = 1)
policy_vi_s_d, v_vi_s_d = value_iteration(env_s, discount_factor = .88)


print("\nOptimal Policy with value iteration (Nothing = 0, 1 floor = 1, 2 floors = 2, 3 floors = 3):")
print(policy_vi,"\n")
print("")

print("\nOptimal Policy with value iteration and discounting (Nothing = 0, 1 floor = 1, 2 floors = 2, 3 floors = 3):")
print(policy_vi_d,"\n")
print("")

print("\nOptimal Policy with value iteration stochastic demand (Nothing = 0, 1 floor = 1, 2 floors = 2, 3 floors = 3):")
print(policy_vi_s,"\n")
print("")

print("\nOptimal Policy with value iteration stochastic demand and discounting (Nothing = 0, 1 floor = 1, 2 floors = 2, 3 floors = 3):")
print(policy_vi_s_d,"\n")
print("")



NPV_vi, model_vi = NPV_garage_DP(policy_vi)
print("NPV for value iteration policy ", NPV_vi)

NPV_vi_d, model_vi_d = NPV_garage_DP(policy_vi)
print("NPV for value iteration policy with discounting", NPV_vi_d)

ENPV_vi = ENPV_MC(1000, policy_vi_s)
ENPV_vi_d = ENPV_MC(1000, policy_vi_s_d)
print("ENPV for optimal policy with stochastic  demand using value iteration policy  is Million $", ENPV_vi*(10**-6))
print("ENPV for optimal policy with stochastic  demand using value iteration policy and discounting is Million $", ENPV_vi*(10**-6))




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
