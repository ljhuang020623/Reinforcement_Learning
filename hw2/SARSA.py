import numpy as np
import matplotlib.pyplot as plt
from pp2starter import prepCartPole, prepFrozen4, prepFrozen8


def choose_action(Q, state, nA, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(nA)
    else:
        best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
        return np.random.choice(best_actions)

def sarsa(env, phi, nS, nA, episodes, max_steps, alpha, gamma, initial_epsilon=0.5, initial_Q_value=0.0):
    Q = np.full((nS, nA), initial_Q_value, dtype=np.float64)
    
    for ep in range(episodes):
        epsilon = initial_epsilon * (episodes - ep - 1) / (episodes - 1)
        
        observation, info = env.reset()
        state = phi(observation)
        action = choose_action(Q, state, nA, epsilon)
        
        for t in range(max_steps):
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = phi(observation)
            if terminated or truncated:
                delta = reward - Q[state, action]
                Q[state, action] += alpha * delta
                break
            else:
                next_action = choose_action(Q, next_state, nA, epsilon)
                delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
                Q[state, action] += alpha * delta
                state, action = next_state, next_action
    return Q

def evaluate_policy(env, phi, Q, num_episodes=500):
    rewards = []
    for _ in range(num_episodes):
        observation, info = env.reset()
        state = phi(observation)
        total_reward = 0
        for _ in range(env._max_episode_steps):
            best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
            action = np.random.choice(best_actions)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = phi(observation)
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards)

def run_experiments(prep_fn, total_episodes, eval_interval, n_repeats, alpha, gamma,
                    initial_epsilon=0.5, initial_Q_value=0.0):
    nS, nA, env, phi, dname = prep_fn()
    max_steps = env._max_episode_steps
    eval_points = list(range(eval_interval, total_episodes + 1, eval_interval))
    all_scores = np.zeros((n_repeats, len(eval_points)))
    
    for rep in range(n_repeats):
        print(f"Starting repeat {rep+1}/{n_repeats}...")
        Q = np.full((nS, nA), initial_Q_value, dtype=np.float64)
        eval_counter = 0
        for ep in range(total_episodes):
            epsilon = initial_epsilon * (total_episodes - ep - 1) / (total_episodes - 1)
            observation, info = env.reset()
            state = phi(observation)
            action = choose_action(Q, state, nA, epsilon)
            
            for t in range(max_steps):
                observation, reward, terminated, truncated, info = env.step(action)
                next_state = phi(observation)
                if terminated or truncated:
                    delta = reward - Q[state, action]
                    Q[state, action] += alpha * delta
                    break
                else:
                    next_action = choose_action(Q, next_state, nA, epsilon)
                    delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
                    Q[state, action] += alpha * delta
                    state, action = next_state, next_action
            
            if (ep + 1) % eval_interval == 0:
                score = evaluate_policy(env, phi, Q)
                all_scores[rep, eval_counter] = score
                eval_counter += 1

        env.close()
    
    means = np.mean(all_scores, axis=0)
    stds = np.std(all_scores, axis=0)
    return dname, eval_points, means, stds

def plot_results(dname, eval_points, means, stds):
    plt.figure()
    plt.errorbar(eval_points, means, yerr=stds, fmt='-o', capsize=5)
    plt.title(f'Performance of SARSA on {dname}')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward (over 500 evaluation episodes)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    alpha = 0.01
    gamma = 0.999
    n_repeats = 10
    initial_epsilon = 0.5
    
    total_episodes_f4 = 50000
    eval_interval_f4 = total_episodes_f4 // 20
    dname, eval_points, means, stds = run_experiments(prepFrozen4, total_episodes_f4, eval_interval_f4, 
                                                       n_repeats, alpha, gamma,
                                                       initial_epsilon=initial_epsilon,
                                                       initial_Q_value=0.0)  
    plot_results(dname, eval_points, means, stds)
    
    total_episodes_f8 = 400000
    eval_interval_f8 = total_episodes_f8 // 20
    dname, eval_points, means, stds = run_experiments(prepFrozen8, total_episodes_f8, eval_interval_f8, 
                                                       n_repeats, alpha, gamma,
                                                       initial_epsilon=initial_epsilon,
                                                       initial_Q_value=0.0)
    plot_results(dname, eval_points, means, stds)
    
    total_episodes_cp = 50000
    eval_interval_cp = total_episodes_cp // 20
    dname, eval_points, means, stds = run_experiments(prepCartPole, total_episodes_cp, eval_interval_cp, 
                                                       n_repeats, alpha, gamma,
                                                       initial_epsilon=initial_epsilon,
                                                       initial_Q_value=0.0)
    plot_results(dname, eval_points, means, stds)