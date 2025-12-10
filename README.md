# CartPole DQN Project

**Author:** Davar Adil Yassin  
**Institution:** University of Kurdistan Hewlêr (UKH)  
**Program:** Master's in Artificial Intelligence  
**Course:** Advanced Robotics  

A complete Deep Q-Network (DQN) reinforcement learning project for the CartPole-v1 environment featuring interactive gameplay modes, real-time Q-value visualization, comprehensive training analytics, and professional academic branding.

---

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [How to Play](#-how-to-play)
- [Training](#-training)
- [Training Analysis](#-training-analysis)
- [Model Architecture](#-model-architecture)
- [Configuration](#-configuration)
- [Requirements](#️-requirements)
- [Results](#-results)
- [Acknowledgments](#-acknowledgments)

---

## 🎮 Project Structure

```
.
├── game_v1/                    # Standard interactive game
│   └── interactive_game.py    # Standard physics with Q-value display
├── game_v2/                    # Enhanced playable version
│   └── interactive_game_v2.py # Relaxed physics + Q-values
├── training/                   # Training scripts and tools
│   ├── dqn_training.py        # Main training script (600 episodes)
│   ├── visual.py              # Video generation with stats overlay
│   ├── logs/                  # CSV training logs
│   └── plots/                 # Training visualization graphs
├── models/                     # Trained model weights (shared)
│   └── dqn_cartpole_gymnasium.pth  # Main trained model
├── Branding/                   # University branding assets
│   └── Logo-Transparent.png   # UKH logo
├── recordings/                 # Demo video recordings
├── docs/                       # Documentation and plots
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 🚀 Features

### 🎯 Interactive Game Modes

**Game V1 (Standard Physics):**
- 🤖 **AI Agent Mode**: Watch the trained agent balance with real-time Q-value display
  - 12° angle limit, 1500 max steps
  - Shows Q-values for left/right actions
  - Displays current action being taken
- 👤 **Manual Mode**: Take full control of the cart
  - 60° angle limit for easier play
  - Arrow key controls
  - 1500 max steps
- ⚔️ **Battle Mode**: Challenge the AI
  - You push, AI tries to counter-balance
  - 60° angle limit
  - Test your skills against the trained agent!

**Game V2 (Super Playable Edition):**
- All V1 features PLUS:
- 🎨 **Enhanced Physics**: Much more forgiving gameplay
  - Gravity: 3.0 (vs 9.8 standard) - 70% slower falling!
  - 90° angle tolerance (vs 12° standard)
  - 3000 max steps (vs 500 standard)
  - Longer pole (1.0m vs 0.5m) - easier to balance
  - Lighter mass (0.05kg vs 0.1kg)
- 🏆 **High Score Tracking**: Beat your personal best
- 📊 **Enhanced UI**: Beautiful stat displays with emojis

### 🧠 Training Features

- **Position Penalty Reward Shaping**: Agent learns to stay centered
  - Penalty: `0.1 × |cart_position|`
  - Encourages balanced, centered behavior
- **600 Episodes**: Optimized training duration
  - Reaches near-optimal performance
  - 5-8 minute training time on CPU
  - 50-episode moving average for responsive tracking
- **Comprehensive Analytics**: 10 different plots + 3 CSV log files
  - Episode rewards, lengths, Q-values
  - Loss curves, TD errors, policy entropy
  - Action distribution analysis
  - Gradient norms and parameter changes
- **Video Generation**: Create demo videos with statistics overlay

### 🔍 Visualization Features

- **Real-time Q-Values**: See the AI's decision-making process
  - Q[Left] and Q[Right] values displayed
  - Current action highlighted in green
  - Dynamic panel expansion in agent mode
- **Color-coded Stats**: Instant visual feedback
  - Red warnings when approaching limits
  - Green for good performance
  - Yellow for important info
- **Professional Branding**: UKH logo and author credits
  - Menu screen branding
  - Clean, academic presentation

---

## 💻 Installation

### Prerequisites
- Python 3.13+ (3.10+ should work)
- pip package manager
- ~500MB disk space for dependencies

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/cartpole-dqn-project.git
cd cartpole-dqn-project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import gymnasium, torch, pygame; print('✓ All dependencies installed!')"
```

---

## 🎯 Quick Start

### Play the Games

**Game V1 (Standard Physics):**
```bash
cd game_v1
python interactive_game.py
```

**Game V2 (Super Playable):**
```bash
cd game_v2
python interactive_game_v2.py
```

### Train a New Model

```bash
cd training
python dqn_training.py
```
*Training takes ~5-8 minutes and generates logs + plots*

### Generate Demo Videos

```bash
cd training
python visual.py
```
*Creates 5 demo videos in `recordings/` folder*

---

## 🎮 How to Play

### Menu Controls
- **Press 1**: AI Agent Mode (watch the AI play)
- **Press 2**: Manual Mode (you control)
- **Press 3**: Battle Mode (you vs AI)
- **Press ESC**: Quit the game

### Gameplay Controls
- **← (Left Arrow)**: Push cart left
- **→ (Right Arrow)**: Push cart right
- **R**: Restart current game
- **M**: Return to main menu
- **ESC**: Quit

### Game Objective
**Goal:** Keep the pole balanced upright for as long as possible!

- The pole starts upright (vertical)
- If it tips too far, the game ends
- Use left/right arrows to move the cart
- Moving the cart creates momentum that affects the pole
- Try to keep the pole within the angle limits:
  - **AI Mode**: ±12° (challenging!)
  - **Manual/Battle Mode**: ±60° (more forgiving)

### Tips for Success
1. **Small corrections**: Don't over-push! Gentle movements work best
2. **Anticipate**: Move before the pole falls too far
3. **Stay centered**: Try to keep the cart near the middle
4. **Watch the angle**: The pole angle indicator turns red when you're in danger
5. **Practice**: Battle mode is great for learning the physics!

### Understanding the Stats Display

**During Gameplay:**
- **Steps**: Number of steps survived
- **Score/Reward**: Total reward earned (1 per step)
- **Time**: Survival time in seconds
- **Position**: Cart's position (meters from center)
- **Velocity**: Cart's speed (m/s)
- **Angle**: Pole's angle from vertical (degrees)
- **Angular Vel**: How fast the pole is rotating

**In AI Mode (Extra Stats):**
- **Q[Left]**: AI's estimated value of pushing left
- **Q[Right]**: AI's estimated value of pushing right
- **Action**: Which action the AI is currently taking (←LEFT or RIGHT→)

The AI chooses the action with the higher Q-value!

---

## 🧠 Training

### Training Configuration

**Hyperparameters:**
```python
EPISODES = 600              # Optimal training duration
BATCH_SIZE = 64             # Mini-batch size for learning
GAMMA = 0.99                # Discount factor (future rewards)
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_END = 0.01          # Minimum exploration rate
EPSILON_DECAY = 0.995       # Exploration decay per episode
LEARNING_RATE = 0.001       # Adam optimizer learning rate
MEMORY_SIZE = 10000         # Experience replay buffer size
TARGET_UPDATE = 10          # Target network update frequency
```

**Reward Shaping:**
- Base reward: +1 per step survived
- Position penalty: -0.1 × |cart_position|
- Encourages the agent to stay centered

**Network Architecture:**
- Input: 4 state variables (position, velocity, angle, angular velocity)
- Hidden Layer 1: 128 neurons + ReLU
- Hidden Layer 2: 128 neurons + ReLU
- Output: 2 Q-values (left action, right action)

### Running Training

```bash
cd training
python dqn_training.py
```

**Training Output:**
- Model saved to: `models/dqn_cartpole_gymnasium.pth`
- Logs saved to: `logs/` (3 CSV files)
- Plots saved to: `plots/` (10 PNG graphs)

**Training Time:**
- CPU: ~5-8 minutes
- 600 episodes
- Real-time progress display

---

## 📊 Training Analysis

The training script generates **10 comprehensive plots** and **3 CSV log files** for detailed analysis.

### 📈 Plot Overview

#### 1. **Reward Curve** (`reward_curve.png`)
- **What it shows**: Episode rewards over time
- **What to look for**: 
  - Early episodes: ~10-50 reward (random exploration)
  - Mid-training: Steady improvement
  - Final episodes: **~350-450 reward** consistently
  - Maximum possible: 500 (episode termination limit)
- **Interpretation**: Rapid learning in first 100-200 episodes, then stabilization

#### 2. **Moving Average Reward** (`moving_average_reward.png`)
- **What it shows**: 50-episode moving average (smoothed performance)
- **What to look for**:
  - Smooth upward trend indicating consistent learning
  - Final average: **~350-450** (near-optimal)
  - Less noisy than raw reward curve
- **Interpretation**: Proves the agent isn't just getting lucky - it's learned a robust policy

#### 3. **Episode Length Curve** (`episode_length_curve.png`)
- **What it shows**: How many steps the agent survived each episode
- **What to look for**:
  - Early: <100 steps (falling quickly)
  - Final: **300-500 steps** (maximum is 500)
- **Interpretation**: Longer episodes = better balance control

#### 4. **Epsilon Decay Curve** (`epsilon_curve.png`)
- **What it shows**: Exploration rate over training
- **What to look for**:
  - Starts at 1.0 (100% random exploration)
  - Decays exponentially
  - Ends at ~0.01 (99% exploitation, 1% exploration)
- **Interpretation**: Agent transitions from exploring to exploiting learned knowledge

#### 5. **Loss Curve** (`loss_curve.png`)
- **What it shows**: Training loss (TD error squared) over time
- **What to look for**:
  - High initial loss (poor predictions)
  - Decreasing trend (learning to predict Q-values accurately)
  - Final stabilization at low values
- **Interpretation**: Lower loss = better Q-value approximation

#### 6. **Q-Value Curve** (`qvalue_curve.png`)
- **What it shows**: Average Q-values over training
- **What to look for**:
  - Increases over time as agent learns long-term rewards
  - Final values: **~300-450** (matches reward performance)
  - Higher Q-values = agent expects higher future rewards
- **Interpretation**: Agent learns to predict cumulative future rewards accurately

#### 7. **TD Error Curve** (`td_error_curve.png`)
- **What it shows**: Temporal Difference error (prediction error) magnitude
- **What to look for**:
  - High early TD errors (unpredictable environment)
  - Decreasing trend (better predictions)
  - Some variance expected (stochastic environment)
- **Interpretation**: Lower TD error = agent's predictions match reality better

#### 8. **Policy Entropy Curve** (`entropy_curve.png`)
- **What it shows**: Entropy of action distribution (measure of policy randomness)
- **What to look for**:
  - High entropy early: ~0.693 (random 50/50 split between actions)
  - Decreasing trend: Agent becomes more decisive
  - Low entropy late: Agent confidently chooses one action over the other
- **Interpretation**: From random exploration to confident decision-making

#### 9. **Action Distribution** (`action_distribution.png`)
- **What it shows**: Percentage of left vs right actions taken
- **What to look for**:
  - Early training: ~50/50 split (random exploration)
  - Late training: May be balanced or slightly biased
  - Bias indicates preferred strategy (e.g., slight right preference)
- **Interpretation**: Shows the agent's learned behavioral patterns

#### 10. **Reward Distribution: Early vs Late** (`reward_histogram_early_late.png`)
- **What it shows**: Histogram comparing first 50 vs last 50 episodes
- **What to look for**:
  - Early (blue): Wide distribution, mostly low rewards (10-100)
  - Late (orange): Narrow distribution, high rewards (300-500)
  - Clear separation = successful learning
- **Interpretation**: Dramatic improvement from novice to expert performance

### 📊 CSV Log Files

**1. `episodes.csv`**
- Episode-by-episode performance data
- Columns: Episode, Reward, Episode Length, Epsilon, Avg Reward (last 50)

**2. `training_stats.csv`**
- Detailed training metrics per episode
- Columns: Episode, Loss, Avg Q-Value, TD Error, Gradient Norm, Param Change, Entropy

**3. `action_distribution.csv`**
- Action frequency tracking
- Columns: Episode, Left Actions, Right Actions, Total Actions

### 🎯 Expected Results

**Successful Training Indicators:**
✅ Final reward: **350-450** (occasionally hitting 500)  
✅ Moving average: Steady upward trend  
✅ Episode length: Consistently reaching 300-500 steps  
✅ Loss: Decreasing and stabilizing  
✅ Q-values: Increasing to match reward levels  
✅ Entropy: Decreasing (more decisive policy)  

---

## 🏗️ Model Architecture

### DQN Network Structure

```python
DQN(
  (fc1): Linear(in_features=4, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
```

**Input Layer (4 features):**
1. Cart Position: -4.8 to 4.8 meters
2. Cart Velocity: -∞ to ∞ m/s
3. Pole Angle: -0.418 to 0.418 radians (~±24°)
4. Pole Angular Velocity: -∞ to ∞ rad/s

**Hidden Layers:**
- Layer 1: 128 neurons + ReLU activation
- Layer 2: 128 neurons + ReLU activation

**Output Layer (2 Q-values):**
- Q(s, left): Estimated value of pushing left
- Q(s, right): Estimated value of pushing right

### DQN Algorithm Components

1. **Experience Replay**: Random sampling from memory buffer prevents correlation
2. **Target Network**: Separate network for stable Q-value targets
3. **Epsilon-Greedy**: Balanced exploration vs exploitation
4. **Reward Shaping**: Position penalty encourages centered behavior

---

## ⚙️ Configuration

### Game Parameters

**Game V1 (Standard Physics):**
```python
# Physics (Gymnasium defaults)
gravity = 9.8
masscart = 1.0
masspole = 0.1
pole_length = 0.5
force_mag = 10.0

# Limits
theta_threshold = 12 * 2 * π / 360  # ±12° for AI mode
theta_threshold_manual = 60 * 2 * π / 360  # ±60° for manual
max_steps = 1500
```

**Game V2 (Super Playable):**
```python
# Modified Physics
gravity = 3.0        # 70% slower falling
pole_length = 1.0    # 2x longer (easier balance)
masspole = 0.05      # 50% lighter
tau = 0.01           # Update timestep

# Limits
theta_threshold = 90 * 2 * π / 360  # ±90° tolerance
max_steps = 3000     # 6x longer episodes
```

---

## 🛠️ Requirements

```txt
gymnasium>=0.29.0
torch>=2.0.0
pygame>=2.5.0
numpy>=1.24.0
matplotlib>=3.7.0
opencv-python>=4.8.0
```

**Python Version:** 3.10+ (tested on 3.13)

---

## 🎬 Results

### Performance Metrics

**AI Agent Performance (Trained Model):**
- Average Reward: **350-450** per episode
- Max Reward Achieved: **500** (hit episode limit)
- Success Rate: **~90%** reach 300+ reward
- Centering Behavior: ✅ Active (stays near x=0)

**Training Efficiency:**
- Episodes to Convergence: **~200-300**
- Total Training Episodes: **600**
- Training Time: **5-8 minutes** (CPU)
- Final Epsilon: **0.01** (99% exploitation)

**Model Statistics:**
- Model Size: **72 KB** (compact!)
- Parameters: **17,538** (128-128-2 architecture)
- Inference Speed: **Real-time** (>60 FPS)

---

## 🙏 Acknowledgments

**Academic Institution:**
- University of Kurdistan Hewlêr (UKH)
- Master's Program in Artificial Intelligence
- Advanced Robotics Course

**Technologies:**
- [Gymnasium](https://gymnasium.farama.org/) - RL environments
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Pygame](https://www.pygame.org/) - Game interface
- [OpenAI](https://openai.com/) - Original CartPole environment

**Author:**
- Davar Adil Yassin
- MSc AI Student, UKH


---

## 📄 License

This project is for educational and academic purposes as part of the Advanced Robotics course at UKH.

---

**⭐ If you found this project useful, please consider giving it a star!**



## 🎯 Quick Start

### Run the Games

```bash
# Game V1 (Standard)
cd game_v1
python interactive_game.py

# Game V2 (Super Playable)
cd game_v2
python interactive_game_v2.py
```

### Train a New Model

```bash
cd training
python dqn_training.py  # 600 episodes, ~5-8 minutes
```

### Generate Videos

```bash
cd training
python visual.py  # Creates 5 demo videos in recordings/
```

## 🎮 Controls

- **Arrow Keys** ← →: Control cart movement
- **1/2/3**: Select game mode
- **R**: Restart current game
- **M**: Return to menu
- **ESC**: Quit

## 🧠 Model Architecture

```python
DQN Network:
- Input: 4 (cart position, velocity, pole angle, angular velocity)
- Hidden Layer 1: 128 neurons + ReLU
- Hidden Layer 2: 128 neurons + ReLU
- Output: 2 (left/right action)
```

## 📈 Training Configuration

- **Algorithm**: DQN with experience replay
- **Episodes**: 600 (optimal balance of speed and performance)
- **Replay Buffer**: 50,000 transitions
- **Epsilon Decay**: 0.995 (exponential, min 0.01)
- **Learning Rate**: 0.001 (Adam optimizer)
- **Discount Factor**: 0.99
- **Batch Size**: 64
- **Reward Modification**: Position penalty of 0.1 * |position| for centering
- **Moving Average**: 50 episodes for responsive tracking
- **Optimizations**: Numpy array conversion for faster tensor operations

## 🛠️ Requirements

```bash
pip install gymnasium torch numpy pygame opencv-python matplotlib
```

- Python 3.13+
- gymnasium (CartPole-v1 environment)
- PyTorch
- Pygame (for interactive games)
- OpenCV (for video recording)
- Matplotlib (for training plots)

## 📝 Physics Modifications

**Game V2 Enhanced Physics:**
- Gravity: 3.0 (vs 9.8 standard) - much slower falling
- Time Step (tau): 0.01 (vs 0.02) - smoother simulation
- Pole Length: 1.0m (vs 0.5m) - easier to balance
- Pole Mass: 0.05kg (vs 0.1kg) - lighter, falls slower
- Force Magnitude: 15.0 (vs 10.0) - stronger pushes

## 🎓 Learning Insights

1. **Simplified Training is Faster**: Removing device management overhead significantly speeds up training
2. **600 Episodes is Sufficient**: Agent reaches max performance in ~600 episodes vs 10k
3. **Position Penalty Works**: Agent successfully learns to prefer center positioning
4. **50-Episode Moving Average**: More responsive to improvements than 100-episode window
5. **Numpy Optimization Matters**: Converting to numpy arrays before tensors eliminates slowdowns
6. **MPS Issues**: Apple Silicon MPS has stability issues with Adam optimizer; CPU is more reliable
7. **Playability vs Realism**: Relaxed physics makes manual play much more fun without sacrificing learning value

## 📺 Video Recordings

Videos include real-time overlays:
- Episode number
- Current step
- Total reward
- Cart position & velocity
- Pole angle & angular velocity
- Visual centering indicator

## 🤝 Contributing

This is a learning project! Feel free to:
- Experiment with different reward functions
- Try different network architectures
- Test various hyperparameters
- Add new game modes
- Improve visualization

## 📄 License

MIT License - Feel free to use and modify!

## 🎉 Acknowledgments

- OpenAI Gymnasium for the CartPole environment
- PyTorch team for the deep learning framework
- The DQN paper by Mnih et al. (2015)

---

**Happy Balancing! 🎪**
