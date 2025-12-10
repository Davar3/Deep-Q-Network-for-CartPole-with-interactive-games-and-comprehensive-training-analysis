import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import os

# --- MUST MATCH YOUR TRAINING CODE EXACTLY ---
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

def visualize():
    print("--- Loading your perfect model ---")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # Create recordings folder if it doesn't exist
    recordings_path = "../recordings"
    os.makedirs(recordings_path, exist_ok=True)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    model = DQN(input_dim, output_dim)
    
    # Load the model from the models folder
    model_path = "../models/dqn_cartpole_gymnasium.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval() # Important: Switch to 'eval' mode
        print(f"✓ Model loaded successfully from: {model_path}")
    except FileNotFoundError:
        print(f"✗ Error: Model not found at '{model_path}'")
        print("Make sure you've trained the model first using training/dqn_training.py")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Run 5 demo episodes
    for episode in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frames = []
        states_history = []
        step_count = 0
        
        print(f"Starting Demo {episode + 1}...")
        
        while not done:
            # Capture frame
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())
            
            # Store state for statistics
            states_history.append(state.copy())
            
            # Prepare state
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # PURE SKILL: No random choices (epsilon), just the best move
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            # Slight delay to make the video look natural (optional)
            time.sleep(0.005)
        
        # Add text overlay to frames
        if frames and states_history:
            # Make sure we have same number of states and frames
            min_len = min(len(frames), len(states_history))
            frames = frames[:min_len]
            states_history = states_history[:min_len]
            
            for i, frame in enumerate(frames):
                frame_with_text = frame.copy()
                frame_bgr = cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR)
                
                # Extract state variables
                cart_position = states_history[i][0]  # Position of cart
                cart_velocity = states_history[i][1]  # Velocity of cart
                pole_angle = states_history[i][2]      # Angle of pole (in radians)
                pole_velocity = states_history[i][3]   # Angular velocity of pole
                
                # Convert angle to degrees for better understanding
                pole_angle_deg = np.degrees(pole_angle)
                
                # Calculate reward for this step (cumulative)
                cumulative_reward = i + 1  # CartPole gives +1 reward per step
                
                # Text to display
                text_lines = [
                    f"Episode: {episode + 1}/5",
                    f"Step: {i + 1}",
                    f"Cart Position: {cart_position:.3f} m",
                    f"Cart Velocity: {cart_velocity:.3f} m/s",
                    f"Pole Angle: {pole_angle_deg:.2f}°",
                    f"Pole Velocity: {pole_velocity:.3f} rad/s",
                    f"Cumulative Reward: {cumulative_reward}",
                    f"Total Episode Reward: {total_reward:.0f}"
                ]
                
                # Add text to frame
                y_offset = 30
                for line in text_lines:
                    cv2.putText(frame_bgr, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 30
                
                # Update frames list with text overlay
                frames[i] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Save video
        if frames:
            video_path = f"{recordings_path}/cartpole_demo_episode_{episode + 1}.mp4"
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Demo {episode + 1} Score: {total_reward}")
            print(f"Video saved: {video_path}")
        else:
            print(f"Demo {episode + 1} Score: {total_reward} (no frames captured)")
        
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    visualize()