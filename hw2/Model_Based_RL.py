import numpy as np
import matplotlib.pyplot as plt
from pp1starter import prepFrozen
from VI_PI_MPI import value_iteration, policy_evaluation
from pp2starter import prepFrozen4, prepFrozen8  

def collect_data(env, N):
    counts = {}            
    transition_counts = {} 
    reward_sums = {}       
    
    for episode in range(N):
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
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
                s_primes = set()
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

def run_random_experiment(env, nS, nA, discount, tolerance, N_values, num_experiments=10):
    results = {N: [] for N in N_values}
    for N in N_values:
        print(f"[Random] Data collection: N = {N}")
        for exp in range(num_experiments):
            counts, transition_counts, reward_sums = collect_data(env, N)
            P_est = estimate_model(counts, transition_counts, reward_sums, nS, nA)
            policy_est, _, _, _ = value_iteration(P_est, nS, nA, discount, tolerance, max_iter=500, env=None)
            score = policy_evaluation(env, policy_est, discount, episodes=500)
            results[N].append(score)
            print(f"  Experiment {exp+1}: Score = {score:.4f}")
    return results


def choose_action(Q, state, nA, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(nA)
    else:
        best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
        return np.random.choice(best_actions)

def build_model(data, nS, nA, variant='standard', M=10, R=1):
    counts = np.zeros((nS, nA, nS))
    rewards = np.zeros((nS, nA, nS))
    sa_counts = np.zeros((nS, nA))
    dead_end = np.zeros(nS, dtype=bool)
    
    for (s, a, r, s_next, terminated) in data:
        if s < nS and s_next < nS:
            counts[s, a, s_next] += 1
            rewards[s, a, s_next] += r
            sa_counts[s, a] += 1
            if terminated:
                dead_end[s_next] = True

    if variant == 'standard' or variant == 'onPi':
        P = np.zeros((nS, nA, nS))
        R_model = np.zeros((nS, nA))
        for s in range(nS):
            for a in range(nA):
                if sa_counts[s, a] > 0:
                    P[s, a, :] = counts[s, a, :] / sa_counts[s, a]
                    R_model[s, a] = np.sum(rewards[s, a, :]) / sa_counts[s, a]
                else:
                    P[s, a, s] = 1.0
                    R_model[s, a] = 0.0
        return P, R_model
    elif variant == 'RMax':
        nS_new = nS + 1  
        P = np.zeros((nS_new, nA, nS_new))
        R_model = np.zeros((nS_new, nA))
        for s in range(nS):
            for a in range(nA):
                if sa_counts[s, a] >= M:
                    P[s, a, :nS] = counts[s, a, :] / sa_counts[s, a]
                    R_model[s, a] = np.sum(rewards[s, a, :]) / sa_counts[s, a]
                else:
                    if not dead_end[s]:
                        P[s, a, nS] = 1.0
                        R_model[s, a] = R
                    else:
                        P[s, a, s] = 1.0
                        R_model[s, a] = 0.0
        for a in range(nA):
            P[nS, a, nS] = 1.0
            R_model[nS, a] = 0.0
        return P, R_model

def value_iteration_model(P, R_model, gamma, threshold=1e-6, max_iter=10000):
    nS, nA, _ = P.shape
    V = np.zeros(nS)
    for _ in range(max_iter):
        V_prev = V.copy()
        for s in range(nS):
            Q_s = np.zeros(nA)
            for a in range(nA):
                Q_s[a] = R_model[s, a] + gamma * np.sum(P[s, a, :] * V_prev)
            V[s] = np.max(Q_s)
        if np.max(np.abs(V - V_prev)) < threshold:
            break
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            Q[s, a] = R_model[s, a] + gamma * np.sum(P[s, a, :] * V)
    return V, Q

def evaluate_policy_model(env, phi, Q, num_episodes=500):
    rewards_eval = []
    for _ in range(num_episodes):
        observation, info = env.reset()
        state = phi(observation)
        total_reward = 0
        done = False
        while not done:
            best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
            action = np.random.choice(best_actions)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = phi(observation)
            done = terminated or truncated
        rewards_eval.append(total_reward)
    return np.mean(rewards_eval)

def model_based_rl(prep_fn, total_episodes, eval_interval, n_repeats,
                   gamma=0.999, initial_epsilon=0.5, variant='onPi', M=10, R=1):
    nS, nA, env, phi, dname = prep_fn()
    quality_all = np.zeros((n_repeats, total_episodes // eval_interval))
    
    for rep in range(n_repeats):
        print(f"[{variant}] Starting repeat {rep+1}/{n_repeats}...")
        data = [] 
        Q = np.zeros((nS, nA))  
        eval_counter = 0
        for ep in range(total_episodes):
            if (ep + 1) % eval_interval == 0:
                P_eval, R_eval = build_model(data, nS, nA, variant='standard')
                _, Q_eval = value_iteration_model(P_eval, R_eval, gamma)
                quality = evaluate_policy_model(env, phi, Q_eval)
                quality_all[rep, eval_counter] = quality
                eval_counter += 1
                print(f"  Episode {ep+1}/{total_episodes}: Eval quality = {quality:.2f}")
            observation, info = env.reset()
            state = phi(observation)
            done = False
            while not done:
                if variant == 'onPi':
                    action = choose_action(Q, state, nA, epsilon=initial_epsilon)
                elif variant == 'RMax':
                    action = np.argmax(Q[state])
                else:
                    raise ValueError("Unknown variant")
                observation, reward, terminated, truncated, info = env.step(action)
                next_state = phi(observation)
                data.append((state, action, reward, next_state, terminated))
                state = next_state
                done = terminated or truncated
        env.close()
    return dname, quality_all

def plot_results(eval_points, quality_all, title):
    mean_quality = np.mean(quality_all, axis=0)
    std_quality = np.std(quality_all, axis=0)
    plt.figure()
    plt.errorbar(eval_points, mean_quality, yerr=std_quality, fmt='-o', capsize=5)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Evaluation Reward')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    env_random, P_random, nS, nA, dname_random = prepFrozen()
    tolerance = 0.001
    discount = 0.999
    N_values = range(2500, 50001, 2500)
    random_results = run_random_experiment(env_random, nS, nA, discount, tolerance, N_values, num_experiments=10)
    
    N_means = []
    N_stds = []
    for N in N_values:
        scores = np.array(random_results[N])
        N_means.append(np.mean(scores))
        N_stds.append(np.std(scores))
    plt.figure(figsize=(8, 6))
    plt.errorbar(list(N_values), N_means, yerr=N_stds, fmt='o-', capsize=5)
    plt.xlabel('Number of Episodes (N)')
    plt.ylabel('Mean Discounted Return')
    plt.title('Random Model Baseline Performance')
    plt.grid(True)
    plt.show()
    

    total_episodes_f4 = 1500
    eval_interval_f4 = total_episodes_f4 // 20
    dname, quality_onPi = model_based_rl(prepFrozen4, total_episodes_f4, eval_interval_f4,
                                         n_repeats=2, gamma=discount,
                                         initial_epsilon=0.5, variant='onPi')
    eval_points_f4 = list(range(eval_interval_f4, total_episodes_f4+1, eval_interval_f4))
    plot_results(eval_points_f4, quality_onPi, "Model-Based RL onPi (FrozenLake4)")
    
    
    dname, quality_RMax = model_based_rl(prepFrozen4, total_episodes_f4, eval_interval_f4,
                                         n_repeats=2, gamma=discount,
                                         initial_epsilon=0.5, variant='RMax', M=10, R=1)
    plot_results(eval_points_f4, quality_RMax, "Model-Based RL RMax (FrozenLake4)")
    
    total_episodes_f8 = 50000
    eval_interval_f8 = total_episodes_f8 // 20
    dname, quality_onPi_f8 = model_based_rl(prepFrozen8, total_episodes_f8, eval_interval_f8,
                                            n_repeats=2, gamma=discount,
                                            initial_epsilon=0.5, variant='onPi')
    eval_points_f8 = list(range(eval_interval_f8, total_episodes_f8+1, eval_interval_f8))
    plot_results(eval_points_f8, quality_onPi_f8, "Model-Based RL onPi (FrozenLake8)")
    
    dname, quality_RMax_f8 = model_based_rl(prepFrozen8, total_episodes_f8, eval_interval_f8,
                                            n_repeats=2, gamma=discount,
                                            initial_epsilon=0.5, variant='RMax', M=10, R=1)
    plot_results(eval_points_f8, quality_RMax_f8, "Model-Based RL RMax (FrozenLake8)")
