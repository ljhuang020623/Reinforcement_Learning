from pp1starter import prepFrozen
from VI_PI_MPI import value_iteration, policy_evaluation
import numpy as np
import matplotlib.pyplot as plt


def collect_data(env, N):
    counts = {} # key: (s,a), value: count, count each (s,a) is encountered
    transition_counts = {}  # key: (s,a,s'), value: count, for each (s,a) count how many times you transition to s'
    reward_sums = {}  # key: (s,a), value: total reward, sum up the reward for each (s,a)

    for episode in range(N):
        # every new episode reset the start position
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            # random action
            action = env.action_space.sample() 
            next_obs, reward, terminated, truncated, info = env.step(action)
            key = (observation, action)
            counts[key] = counts.get(key, 0) + 1
            transition_key = (observation, action, next_obs)
            transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
            reward_sums[key] = reward_sums.get(key, 0) + reward
            observation = next_obs
    return counts, transition_counts, reward_sums

def estimate_model(counts, transition_counts, reward_sums, nS, nA):
    # initialize estimate model with dimensions [nS][nA]
    P_est = []
    for s in range(nS):
        action_list = []
        for a in range(nA):
            action_list.append(None)
        P_est.append(action_list)

    for s in range(nS):
        for a in range(nA):
            key = (s, a)
            total = counts.get(key, 0)
            outcomes = []
            if total > 0:
                s_primes  = set()
                for s0, a0, s_ in transition_counts.keys():
                    if s0 == s and a0 == a:
                        s_primes.add(s_)
                for s_prime in s_primes:
                    transition_key = (s, a, s_prime)
                    count_sas = transition_counts.get(transition_key, 0)
                    p_est = count_sas / total
                    r_est = reward_sums.get(key, 0) / total
                    outcomes.append((p_est, s_prime, r_est, False))
                P_est[s][a] = outcomes
            else:
                P_est[s][a] = [(1.0, s, 0.0, False)]
    return P_est

def run_mbrl_experiment(env, nS, nA, discount, tolerance, N_values, num_experiments=10):
    results = {N: [] for N in N_values}
    for N in N_values:
        print(f"N = {N}")
        for exp in range(num_experiments):
            counts, transition_counts, reward_sums = collect_data(env, N)
            P_est = estimate_model(counts, transition_counts, reward_sums, nS, nA)
            # Plan with VI using the estimated model.
            policy_est, _, _, _ = value_iteration(P_est, nS, nA, discount, tolerance, max_iter=500, env=None)
            # Evaluate the policy on the true model/environment.
            score = policy_evaluation(env, policy_est, discount, episodes=500)
            results[N].append(score)
            print(f"  Experiment {exp+1}: Score = {score:.4f}")
    return results



if __name__ == '__main__':
    env, P, nS, nA, dname = prepFrozen()
    tolerance = 0.001
    discount = 0.999
    
    # Define values for N (number of episodes used for data collection)
    N_values = range(2500, 50001, 2500)
    mbrl_results = run_mbrl_experiment(env, nS, nA, discount, tolerance, N_values, num_experiments=10)
    
    # Compute means and standard deviations.
    N_means = []
    N_stds = []
    for N in N_values:
        scores = np.array(mbrl_results[N])
        N_means.append(np.mean(scores))
        N_stds.append(np.std(scores))
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(list(N_values), N_means, yerr=N_stds, fmt='o-', capsize=5)
    plt.xlabel('Number of Episodes (N)')
    plt.ylabel('Mean Discounted Return')
    plt.title('MBRL Performance vs. Data Size')
    plt.grid(True)
    plt.show()