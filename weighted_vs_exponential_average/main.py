import matplotlib.pyplot as plt
import numpy as np



def run_bandit_experiment(
    k: int =10, runs: int =2000, steps: int =10000, epsilon: float =0.1, alpha: float =0.1, drift_std: float =0.01
) -> dict[str, float]:
    # Initialize: true action values for all runs
    true_q = np.zeros((runs, k))
    optimal_actions = np.zeros((runs, steps))
    rewards_sample_avg = np.zeros((runs, steps))
    rewards_const_alpha = np.zeros((runs, steps))

    # For sample-average method
    Q_sample = np.zeros((runs, k))
    N_sample = np.zeros((runs, k))

    # For constant-alpha method
    Q_const = np.zeros((runs, k))

    for t in range(steps):
        # --- Random walk for nonstationarity ---
        true_q += np.random.normal(0, drift_std, size=(runs, k))
        

        # --- Optimal action (ground truth) ---
        optimal_action: np.float64 = np.argmax(true_q, axis=1)

        # ε-greedy action selection for sample-average
        explore = np.random.rand(runs) < epsilon
        rand_actions = np.random.randint(0, k, size=runs)
        greedy_actions_sample : np.float64  = np.argmax(Q_sample, axis=1)
        actions_sample = np.where(explore, rand_actions, greedy_actions_sample)
        

        # ε-greedy for constant-α
        greedy_actions_const : np.float64 = np.argmax(Q_const, axis=1)
        actions_const = np.where(explore, rand_actions, greedy_actions_const)

        # Gather rewards
        rewards_s = np.random.normal(true_q[np.arange(runs), actions_sample], 1)
        rewards_c = np.random.normal(true_q[np.arange(runs), actions_const], 1)

        # Track optimal actions
        optimal_actions[:, t] = (actions_sample == optimal_action)

        # --- Update estimates: Sample-Average ---
        N_sample[np.arange(runs), actions_sample] += 1
        step_size = 1 / N_sample[np.arange(runs), actions_sample]
        Q_sample[np.arange(runs), actions_sample] += step_size * (
            rewards_s - Q_sample[np.arange(runs), actions_sample]
        )

        # --- Update estimates: Constant Alpha ---
        Q_const[np.arange(runs), actions_const] += alpha * (
            rewards_c - Q_const[np.arange(runs), actions_const]
        )

        # Log rewards
        rewards_sample_avg[:, t] = rewards_s
        rewards_const_alpha[:, t] = rewards_c

    # Return averaged stats over runs
    return {
        "avg_rewards_sample": rewards_sample_avg.mean(axis=0),
        "avg_rewards_const": rewards_const_alpha.mean(axis=0),
        "optimal_action_sample": optimal_actions.mean(axis=0)
    }


results = run_bandit_experiment()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results["avg_rewards_sample"], label="Sample Average")
plt.plot(results["avg_rewards_const"], label="Constant Step-size")
plt.ylabel("Average reward")
plt.xlabel("Steps")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results["optimal_action_sample"] * 100)
plt.ylabel("% Optimal action")
plt.xlabel("Steps")

plt.tight_layout()
plt.savefig("ex.png")
