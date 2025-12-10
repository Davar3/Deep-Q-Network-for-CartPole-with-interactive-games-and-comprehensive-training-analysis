import os
import csv
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Ensure folders exist
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("../models", exist_ok=True)  # Ensure models folder exists

# ---------------------------
# 1. DQN Network
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------
# 2. Agent with stats
# ---------------------------
class Agent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.lr = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.batch_size = 64

        self.eps = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.01
        self.action_dim = action_dim

    def remember(self, transition):
        # transition: (state, action, reward, next_state, done)
        self.memory.append(transition)

    def choose_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state)).item()

    def train(self):
        """
        Returns a dict of training stats for this update, or None
        if not enough samples in replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        dones       = torch.FloatTensor(dones)

        # Current Q estimates
        q_pred = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            q_next = self.target_model(next_states).max(1)[0]
            q_target = rewards + (1 - dones) * self.gamma * q_next

        # TD error
        td_errors = q_target - q_pred

        loss = nn.MSELoss()(q_pred, q_target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient norm (L2 over all params)
        grad_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.data
                grad_norm_sq += g.norm(2).item() ** 2
        grad_norm = grad_norm_sq ** 0.5

        # Save params before update for param-change metric
        params_before = [p.data.clone() for p in self.model.parameters()]

        self.optimizer.step()

        # Parameter change (L2)
        param_change_sq = 0.0
        with torch.no_grad():
            for p, p_before in zip(self.model.parameters(), params_before):
                diff = (p.data - p_before)
                param_change_sq += diff.norm(2).item() ** 2
        param_change = param_change_sq ** 0.5

        # Policy entropy using softmax over Q-values
        with torch.no_grad():
            q_values = self.model(states)  # [batch, actions]
            probs = torch.softmax(q_values, dim=1)
            log_probs = torch.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(dim=1).mean().item()

        stats = {
            "loss": loss.item(),
            "td_error": td_errors.abs().mean().item(),
            "q_value_mean": q_pred.mean().item(),
            "entropy": entropy,
            "grad_norm": grad_norm,
            "param_change": param_change
        }
        return stats

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# ---------------------------
# 3. Helper: write CSV
# ---------------------------
def write_csv(path, header, rows):
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------
# 4. Main Training Loop
# ---------------------------
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = Agent(state_dim, action_dim)

episodes = 600
SUCCESS_THRESHOLD = 500  # define "success" as full 500 steps

# --- per-episode stats ---
episode_rewards = []
episode_lengths = []
moving_avg_rewards = []
epsilons = []
success_flags = []
cumulative_rewards = []
reward_variances = []
length_variances = []
best_10_averages = []
buffer_fullness = []

# training stats (episode-averaged)
mean_losses = []
mean_td_errors = []
mean_q_values = []
mean_entropies = []
mean_grad_norms = []
mean_param_changes = []

# action distribution stats per episode
episode_left_counts = []
episode_right_counts = []
episode_left_fracs = []
episode_right_fracs = []

total_cumulative_reward = 0.0

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    # Per-episode accumulators for training stats
    losses_this_ep = []
    td_errors_this_ep = []
    q_values_this_ep = []
    entropies_this_ep = []
    grad_norms_this_ep = []
    param_changes_this_ep = []

    # Action counts
    left_count = 0
    right_count = 0

    while not (terminated or truncated):
        action = agent.choose_action(state)

        if action == 0:
            left_count += 1
        else:
            right_count += 1

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Reward shaping: penalize distance from center
        cart_position = next_state[0]
        position_penalty = abs(cart_position) * 0.1
        modified_reward = reward - position_penalty

        agent.remember((state, action, modified_reward, next_state, float(terminated)))
        stats = agent.train()
        if stats is not None:
            losses_this_ep.append(stats["loss"])
            td_errors_this_ep.append(stats["td_error"])
            q_values_this_ep.append(stats["q_value_mean"])
            entropies_this_ep.append(stats["entropy"])
            grad_norms_this_ep.append(stats["grad_norm"])
            param_changes_this_ep.append(stats["param_change"])

        state = next_state
        total_reward += reward        # track ORIGINAL env reward
        steps += 1

    # Update target network
    agent.update_target()

    # Epsilon decay
    if agent.eps > agent.eps_min:
        agent.eps *= agent.eps_decay

    # Track episode-level metrics
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    total_cumulative_reward += total_reward
    cumulative_rewards.append(total_cumulative_reward)

    # Moving average of rewards (window = 50)
    window = 50
    if len(episode_rewards) < window:
        mav = np.mean(episode_rewards)
    else:
        mav = np.mean(episode_rewards[-window:])
    moving_avg_rewards.append(mav)

    # Reward & length variance (over all episodes so far)
    reward_variances.append(np.var(episode_rewards))
    length_variances.append(np.var(episode_lengths))

    # Best 10-episode average so far
    if len(episode_rewards) >= 10:
        best_10 = max(
            np.mean(episode_rewards[i:i + 10])
            for i in range(len(episode_rewards) - 9)
        )
    else:
        best_10 = np.mean(episode_rewards)
    best_10_averages.append(best_10)

    # Success flag
    success_flags.append(int(total_reward >= SUCCESS_THRESHOLD))

    # Epsilon log
    epsilons.append(agent.eps)

    # Replay buffer fullness
    buffer_fullness.append(len(agent.memory) / agent.memory.maxlen)

    # Episode-averaged training stats
    def safe_mean(x):
        return float(np.mean(x)) if len(x) > 0 else float("nan")

    mean_losses.append(safe_mean(losses_this_ep))
    mean_td_errors.append(safe_mean(td_errors_this_ep))
    mean_q_values.append(safe_mean(q_values_this_ep))
    mean_entropies.append(safe_mean(entropies_this_ep))
    mean_grad_norms.append(safe_mean(grad_norms_this_ep))
    mean_param_changes.append(safe_mean(param_changes_this_ep))

    # Action distributions this episode
    total_actions = left_count + right_count if (left_count + right_count) > 0 else 1
    left_frac = left_count / total_actions
    right_frac = right_count / total_actions

    episode_left_counts.append(left_count)
    episode_right_counts.append(right_count)
    episode_left_fracs.append(left_frac)
    episode_right_fracs.append(right_frac)

    print(
        f"Episode {ep+1}/{episodes} | "
        f"Reward: {total_reward:.1f} | "
        f"Length: {steps} | "
        f"Moving Avg: {mav:.2f} | "
        f"Eps: {agent.eps:.3f}"
    )

# Save final model to models folder
torch.save(agent.model.state_dict(), "../models/dqn_cartpole_gymnasium.pth")
print("\n" + "="*60)
print("✓ Model saved to ../models/dqn_cartpole_gymnasium.pth")
print("="*60)

env.close()

# ---------------------------
# 5. Write CSV Logs
# ---------------------------
episode_indices = list(range(1, episodes + 1))

# episodes.csv: main per-episode stats
episode_rows = []
for i in range(episodes):
    episode_rows.append([
        episode_indices[i],
        episode_rewards[i],
        episode_lengths[i],
        moving_avg_rewards[i],
        epsilons[i],
        success_flags[i],
        cumulative_rewards[i],
        reward_variances[i],
        length_variances[i],
        best_10_averages[i],
        buffer_fullness[i]
    ])

write_csv(
    "logs/episodes.csv",
    header=[
        "episode",
        "reward",
        "length",
        "moving_avg_reward",
        "epsilon",
        "success",
        "cumulative_reward",
        "reward_variance",
        "length_variance",
        "best_10_avg_reward",
        "replay_buffer_fullness"
    ],
    rows=episode_rows
)

# training_stats.csv: episode-averaged training metrics
training_rows = []
for i in range(episodes):
    training_rows.append([
        episode_indices[i],
        mean_losses[i],
        mean_td_errors[i],
        mean_q_values[i],
        mean_entropies[i],
        mean_grad_norms[i],
        mean_param_changes[i]
    ])

write_csv(
    "logs/training_stats.csv",
    header=[
        "episode",
        "mean_loss",
        "mean_td_error",
        "mean_q_value",
        "mean_entropy",
        "mean_grad_norm",
        "mean_param_change"
    ],
    rows=training_rows
)

# action_distribution.csv
action_rows = []
for i in range(episodes):
    action_rows.append([
        episode_indices[i],
        episode_left_counts[i],
        episode_right_counts[i],
        episode_left_fracs[i],
        episode_right_fracs[i]
    ])

write_csv(
    "logs/action_distribution.csv",
    header=[
        "episode",
        "left_count",
        "right_count",
        "left_fraction",
        "right_fraction"
    ],
    rows=action_rows
)

# ---------------------------
# 6. Plotting (Graphs A–J)
# ---------------------------

# A: Reward vs Episode
plt.figure(figsize=(10, 5))
plt.plot(episode_indices, episode_rewards, label="Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("plots/reward_curve.png")
plt.close()

# B: Episode Length vs Episode (+ trendline with "CI"-style band)
plt.figure(figsize=(10, 5))
plt.plot(episode_indices, episode_lengths, label="Episode Length", alpha=0.6)

# moving avg + std for length
length_ma = []
length_std = []
window = 50
for i in range(episodes):
    start = max(0, i - window + 1)
    segment = episode_lengths[start:i + 1]
    length_ma.append(np.mean(segment))
    length_std.append(np.std(segment))

length_ma = np.array(length_ma)
length_std = np.array(length_std)
plt.plot(episode_indices, length_ma, label="Length Moving Avg (50)", color="black")
plt.fill_between(
    episode_indices,
    length_ma - length_std,
    length_ma + length_std,
    alpha=0.2,
    label="±1 Std Dev"
)

plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Episode Length vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("plots/episode_length_curve.png")
plt.close()

# C: Moving Average Reward
plt.figure(figsize=(10, 5))
plt.plot(episode_indices, moving_avg_rewards, label="Moving Avg Reward (50 episodes)")
plt.xlabel("Episode")
plt.ylabel("Reward (moving avg)")
plt.title("Moving Average Reward vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("plots/moving_average_reward.png")
plt.close()

# D: Epsilon vs Episode
plt.figure(figsize=(10, 5))
plt.plot(episode_indices, epsilons, label="Epsilon")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Exploration Rate (Epsilon) vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("plots/epsilon_curve.png")
plt.close()

# E: Loss vs Episode (mean loss)
plt.figure(figsize=(10, 5))
plt.plot(episode_indices, mean_losses, label="Mean Loss per Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("plots/loss_curve.png")
plt.close()

# F: Q-value stability graph (mean Q)
plt.figure(figsize=(10, 5))
plt.plot(episode_indices, mean_q_values, label="Mean Q-value per Episode")
plt.xlabel("Episode")
plt.ylabel("Q-value")
plt.title("Q-value Stability vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("plots/qvalue_curve.png")
plt.close()

# G: Action Distribution (total across all episodes)
total_left = sum(episode_left_counts)
total_right = sum(episode_right_counts)
plt.figure(figsize=(6, 6))
plt.bar(["Left", "Right"], [total_left, total_right])
plt.title("Total Action Distribution (Left vs Right)")
plt.ylabel("Count")
plt.savefig("plots/action_distribution.png")
plt.close()

# H: Reward Histogram (early vs late)
plt.figure(figsize=(10, 5))
num_hist = min(100, episodes)
early_rewards = episode_rewards[:num_hist]
late_rewards = episode_rewards[-num_hist:] if episodes >= num_hist else episode_rewards

plt.hist(early_rewards, bins=20, alpha=0.5, label=f"Early (first {num_hist} eps)")
plt.hist(late_rewards, bins=20, alpha=0.5, label=f"Late (last {num_hist} eps)")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution: Early vs Late Training")
plt.legend()
plt.grid(True)
plt.savefig("plots/reward_histogram_early_late.png")
plt.close()

# I: Policy Entropy vs Episode
plt.figure(figsize=(10, 5))
plt.plot(episode_indices, mean_entropies, label="Mean Policy Entropy")
plt.xlabel("Episode")
plt.ylabel("Entropy")
plt.title("Policy Entropy vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("plots/entropy_curve.png")
plt.close()

# J: TD Error vs Episode
plt.figure(figsize=(10, 5))
plt.plot(episode_indices, mean_td_errors, label="Mean TD Error")
plt.xlabel("Episode")
plt.ylabel("TD Error")
plt.title("TD Error vs Episode")
plt.legend()
plt.grid(True)
plt.savefig("plots/td_error_curve.png")
plt.close()

print("Training finished. Logs written to 'logs/' and plots written to 'plots/'.")
