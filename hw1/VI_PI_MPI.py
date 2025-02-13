from pp1starter import prepFrozen
import numpy as np
import matplotlib.pyplot as plt


# Policy evaluation function
def policy_evaluation(env, policy, discount, episodes = 500):
    scores = []
    for info in range(episodes):
        # reset every episodes
        observation, info = env.reset()
        terminated = False
        total_reward = 0
        t = 0
        while not terminated:
            action = int(policy[observation])
            # Take the action in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            # Accumulatet the discounted reward
            total_reward += (discount ** t) * reward
            # t represent each steps
            t += 1
            if terminated or truncated:
                break
        scores.append(total_reward)
    return np.mean(scores)


def value_iteration(P, nS, nA, discount, tolerance, max_iter = 500, env = None, eval_every = 10):
    # Iterative Updates:
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    eval_scores = []
    eval_iterations = []

    for iteration in range(max_iter):
        V_old = V.copy()
        for s in range(nS):
            q = np.zeros(nA)
            for a in range(nA):
                for prob, s_, R, done in P[s][a]:
                    q[a] += prob * (R + discount * V_old[s_])
            V[s] = max(q)
            policy[s] = np.argmax(q)

            
        if env is not None and iteration % 10 == 0:
            score = policy_evaluation(env, policy, discount)
            eval_scores.append(score)
            eval_iterations.append(iteration)
            print(f"Iteration {iteration}: Mean return = {score}")

        # Check if it converges
        if np.abs(V - V_old).max() < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

    return policy, V, eval_iterations, eval_scores

def policy_iteration(P, nS, nA, discount, tolerance, max_policy_eval_iter = 500, env = None):
    policy = np.zeros(nS, dtype=int)
    V = np.zeros(nS)
    eval_scores = []
    eval_iterations = []
    backups_count_list = []
    cumulative_backups = 0

    policy_stable = False
    iteration = 0

    while not policy_stable:
        # ---Policy iteration phase---
        for i in range(max_policy_eval_iter):
            V_old = V.copy()
            for s in range(nS):
                a = policy[s]
                V[s] = 0
                for prob, s_, R, done in P[s][a]:
                    V[s] += (prob * (R + discount * V_old[s_]))
                # Count nS backups for this inner iteration
            cumulative_backups += nS
            if np.max(np.abs(V - V_old)) < tolerance:
                break

        # ---Policy evaluation phase---
        if env is not None :
            score = policy_evaluation(env, policy, discount)
            eval_scores.append(score)
            eval_iterations.append(iteration)
            print(f"Iteration {iteration}: Mean return = {score}")
        backups_count_list.append(cumulative_backups)

        # ---Policy improvement phase---
        policy_stable = True
        backups_policy_improvement = 0
        for s in range(nS):
            q = np.zeros(nA)
            old_action = policy[s]
            for a in range(nA):
                for prob, s_, R, done in P[s][a]:
                    q[a] += prob * (R + discount * V[s_])
                # Count one backup per (s, a) pair
                backups_policy_improvement += 1 
            best_action = np.argmax(q)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        cumulative_backups += backups_policy_improvement
        iteration += 1

        if policy_stable:
            print(f"Policy converged after {iteration} iterations")
            break
    return policy, V, eval_iterations, eval_scores, backups_count_list


def modified_policy_iteration(P, nS, nA, discount, tolerance, m=3, env = None):
    policy = np.random.randint(0, nA, size = nS)
    V = np.zeros(nS)
    eval_scores = []
    eval_iterations = []
    backups_count_list = []
    cumulative_backups = 0

    policy_stable = False
    iteration = 0

    while not policy_stable:
        # ---Partial policy iteration phase---
        for i in range(m):
            V_old = V.copy()
            for s in range(nS):
                a = policy[s]
                V[s] = 0
                for prob, s_, R, done in P[s][a]:
                    V[s] += prob * (R + discount * V_old[s_])
            # Count nS backups for this inner iteration
            cumulative_backups += nS

        # ---Policy evaluation phase---
        if env is not None:
            score = policy_evaluation(env, policy, discount)
            eval_scores.append(score)
            eval_iterations.append(iteration)
            print(f"MPI Iteration {iteration}: Mean return = {score:.4f}")
        backups_count_list.append(cumulative_backups)

        # ---Policy improvement phase---
        policy_stable = True
        backups_policy_improvement = 0
        for s in range(nS):
            q = np.zeros(nA)
            old_action = policy[s]
            for a in range(nA):
                for prob, s_, R, done in P[s][a]:
                    q[a] += prob * (R + discount * V[s_])
                backups_policy_improvement += 1  # one backup for each (s, a)
            best_action = np.argmax(q)
            if best_action != old_action:
                policy_stable = False
            policy[s] = best_action
        cumulative_backups += backups_policy_improvement
        iteration += 1
        if np.max(np.abs(V - V_old)) < tolerance:
            print(f"MPI Converged after {iteration} iterations")
            break

    return policy, V, eval_iterations, eval_scores, backups_count_list

if __name__ == '__main__':
    # Create an 8x8 Frozen Lake
    env, P, nS, nA, dname = prepFrozen()
    tolerance = 0.001
    discount = 1.0 - 1E-3  # 0.999

    # ======================
    # Value Iteration (VI)
    print("\n--- Value Iteration (VI) ---")
    vi_policy, vi_V, vi_eval_iterations, vi_eval_scores = value_iteration(P, nS, nA, discount, tolerance, max_iter=500, env=env, eval_every=10)
    
    print("\nFinal VI Policy (reshaped to 8x8):")
    print(vi_policy.reshape(8,8))
    np.set_printoptions(precision=4, suppress=True)
    print("Final VI Value Function (reshaped to 8x8):")
    # Note: Use the .shape attribute (without parentheses) or print the array.
    print(vi_V.reshape(8,8))
    
    # Plot for VI
    plt.figure(figsize=(12, 5))
    # Plot 1: Quality vs. Iterations (VI)
    plt.subplot(1, 2, 1)
    plt.plot(vi_eval_iterations, vi_eval_scores, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Return')
    plt.title('VI: Policy Quality vs. Iterations')
    # Plot 2: Quality vs. Single Action Backups (VI)
    # For VI, each iteration does nS*nA backups.
    vi_backups = [iteration * (nS * nA) for iteration in vi_eval_iterations]
    plt.subplot(1, 2, 2)
    plt.plot(vi_backups, vi_eval_scores, marker='o')
    plt.xlabel('Single Action Backups')
    plt.ylabel('Mean Return')
    plt.title('VI: Policy Quality vs. Single Action Backups')
    plt.tight_layout()
    plt.show()

    # ======================
    # Policy Iteration (PI)
    print("\n--- Policy Iteration (PI) ---")
    pi_policy, pi_V, pi_eval_iterations, pi_eval_scores, pi_backups_count_list = policy_iteration(P, nS, nA, discount, tolerance, max_policy_eval_iter=500, env=env)
    
    print("\nFinal PI Policy (reshaped to 8x8):")
    print(pi_policy.reshape(8,8))
    np.set_printoptions(precision=4, suppress=True)
    print("Final PI Value Function (reshaped to 8x8):")
    print(pi_V.reshape(8,8))
    
    # Plot for PI
    plt.figure(figsize=(12, 5))
    # Plot 1: Policy Quality vs. Policy Iterations (PI)
    plt.subplot(1, 2, 1)
    plt.plot(pi_eval_iterations, pi_eval_scores, marker='o')
    plt.xlabel('Policy Iteration')
    plt.ylabel('Mean Return')
    plt.title('PI: Policy Quality vs. Policy Iterations')
    # Plot 2: Policy Quality vs. Single Action Backups (PI)
    plt.subplot(1, 2, 2)
    plt.plot(pi_backups_count_list, pi_eval_scores, marker='o')
    plt.xlabel('Single Action Backups')
    plt.ylabel('Mean Return')
    plt.title('PI: Policy Quality vs. Single Action Backups')
    plt.tight_layout()
    plt.show()

    
    # ======================
    # Modified Policy Iteration (MPI)
    print("\n--- Modified Policy Iteration (MPI) ---")
    mpi_policy, mpi_V, mpi_eval_iterations, mpi_eval_scores, mpi_backups_count_list = modified_policy_iteration(
        P, nS, nA, discount, tolerance, m=3, env=env)

    print("\nFinal MPI Policy (reshaped to 8x8):")
    print(mpi_policy.reshape(8, 8))
    print("Final MPI Value Function (reshaped to 8x8):")
    print(mpi_V.reshape(8, 8))

    # Plot for MPI
    plt.figure(figsize=(12, 5))
    # Plot 1: Quality vs. MPI Iterations
    plt.subplot(1, 2, 1)
    plt.plot(mpi_eval_iterations, mpi_eval_scores, marker='o', linestyle='-')
    plt.xlabel('MPI Iteration')
    plt.ylabel('Mean Return')
    plt.title('MPI: Quality vs. Iterations')
    # Plot 2: Quality vs. Single Action Backups
    plt.subplot(1, 2, 2)
    plt.plot(mpi_backups_count_list, mpi_eval_scores, marker='o', linestyle='-')
    plt.xlabel('Single Action Backups')
    plt.ylabel('Mean Return')
    plt.title('MPI: Quality vs. Backups')
    plt.tight_layout()
    plt.show()