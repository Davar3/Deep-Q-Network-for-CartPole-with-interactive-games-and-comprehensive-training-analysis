import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import numpy as np
import pygame
import sys
import os

"""
CartPole DQN Interactive Game
Author: Davar Adil Yassin
Master's Student in AI - University of Kurdistan Hewlêr (UKH)
Advanced Robotics Class Assignment
"""

# Custom CartPole wrapper with relaxed termination for manual play
class RelaxedCartPole:
    """Wrapper for CartPole with more forgiving termination conditions"""
    def __init__(self, angle_limit=60.0, position_limit=4.0, max_steps=1500):
        self.env = gym.make("CartPole-v1")
        # Modify physics to slow down pole falling
        self.env.unwrapped.gravity = 7.8  # Default is 9.8, reduced by ~20%
        self.env.unwrapped.tau = 0.025  # Default is 0.02, increased for slower updates
        
        self.angle_limit = np.radians(angle_limit)  # Convert degrees to radians
        self.position_limit = position_limit
        self.max_steps = max_steps
        self.current_step = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Apply custom termination conditions (much more forgiving)
        cart_position = state[0]
        pole_angle = state[2]
        
        # Terminate only if pole falls too far or cart goes way off screen
        terminated = (abs(pole_angle) > self.angle_limit or 
                     abs(cart_position) > self.position_limit)
        
        # Override the default 500 step truncation - use our max_steps instead
        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False  # Ignore the base environment's 500 step limit
        
        return state, reward, terminated, truncated, info
    
    def close(self):
        self.env.close()

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

class InteractiveCartPole:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        
        # Screen dimensions
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Interactive CartPole - Control with Arrow Keys!")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 100, 255)
        self.YELLOW = (255, 255, 0)
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 28)
        self.tiny_font = pygame.font.Font(None, 20)
        
        # Load UKH Logo
        try:
            logo_path = "/Users/davar/Desktop/Robotics Project/new vs code/Branding/Logo-Transparent.png"
            self.ukh_logo = pygame.image.load(logo_path)
            # Scale logo to reasonable size (height = 80px)
            logo_height = 80
            aspect_ratio = self.ukh_logo.get_width() / self.ukh_logo.get_height()
            logo_width = int(logo_height * aspect_ratio)
            self.ukh_logo = pygame.transform.scale(self.ukh_logo, (logo_width, logo_height))
            self.logo_loaded = True
        except:
            print("Warning: UKH logo not found.")
            self.logo_loaded = False
        
        # Game state
        self.mode = "menu"  # "menu", "agent", "manual", "both"
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Load environment (will be recreated based on mode)
        self.env = None
        self.create_environment("agent")  # Start with standard env
        
        # Load DQN model
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.model = DQN(input_dim, output_dim)
        
        # Use absolute path to models folder
        model_path = "/Users/davar/Desktop/Robotics Project/new vs code/models/dqn_cartpole_gymnasium.pth"
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            self.model.eval()
            print(f"✓ Model loaded successfully from: {model_path}")
            self.model_loaded = True
        except FileNotFoundError:
            print(f"✗ Warning: Model not found at {model_path}. Agent control disabled.")
            self.model_loaded = False
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model_loaded = False
        
        # Game variables
        self.total_reward = 0
        self.steps = 0
        self.manual_action = None
        self.game_over = False
        self.q_values = None  # Store Q-values for debugging
    
    def create_environment(self, mode):
        """Create appropriate environment based on mode"""
        if self.env is not None:
            self.env.close()
        
        if mode in ["manual", "both"]:
            # Use relaxed CartPole for manual modes (60 degrees instead of 12, 1500 steps)
            self.env = RelaxedCartPole(angle_limit=60.0, position_limit=4.0, max_steps=1500)
        else:
            # Use standard CartPole for AI mode (but extend to 1500 steps)
            self.env = RelaxedCartPole(angle_limit=12.0, position_limit=2.4, max_steps=1500)
        
        self.state, _ = self.env.reset()
        
    def draw_text(self, text, x, y, color=None, font=None):
        if color is None:
            color = self.WHITE
        if font is None:
            font = self.font
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
    
    def draw_menu(self):
        self.screen.fill(self.BLACK)
        
        title = "Interactive CartPole Game"
        self.draw_text(title, self.screen_width // 2 - 250, 100, self.GREEN, pygame.font.Font(None, 64))
        
        # Display author info and logo
        if self.logo_loaded:
            logo_x = self.screen_width - self.ukh_logo.get_width() - 30
            logo_y = 20
            self.screen.blit(self.ukh_logo, (logo_x, logo_y))
        
        self.draw_text("By: Davar Adil Yassin", 50, 180, self.WHITE, self.small_font)
        self.draw_text("Master's Student in AI - UKH", 50, 210, self.WHITE, self.tiny_font)
        self.draw_text("Advanced Robotics Class", 50, 230, self.WHITE, self.tiny_font)
        
        self.draw_text("Select Mode:", self.screen_width // 2 - 100, 280)
        self.draw_text("Press 1: AI Agent Mode (Watch AI balance - max 1500 steps)", 50, 350, self.BLUE)
        self.draw_text("Press 2: Manual Mode (100% YOU control - 60° / 1500 steps)", 50, 420, self.YELLOW)
        self.draw_text("Press 3: Battle Mode (You push, AI fights back - 60°)", 50, 490, self.GREEN)
        self.draw_text("Press ESC: Quit", 50, 560, self.RED)
        
        instructions = [
            "",
            "How to play:",
            "- Use LEFT/RIGHT arrow keys to apply force",
            "- Try to keep the pole upright!",
            "- Press R to restart anytime",
            "- Press M to return to menu"
        ]
        
        y_offset = 630
        for line in instructions:
            self.draw_text(line, 50, y_offset, self.WHITE, self.small_font)
            y_offset += 30
    
    def draw_cartpole(self):
        # Extract state
        cart_position = self.state[0]
        pole_angle = self.state[2]
        
        # Scale for display
        scale = 100
        cart_x = self.screen_width // 2 + int(cart_position * scale)
        cart_y = self.screen_height // 2 + 100
        cart_width = 60
        cart_height = 40
        
        # Draw track
        pygame.draw.line(self.screen, self.WHITE, (50, cart_y + cart_height // 2), 
                        (self.screen_width - 50, cart_y + cart_height // 2), 3)
        
        # Draw center line
        center_x = self.screen_width // 2
        pygame.draw.line(self.screen, self.GREEN, (center_x, cart_y - 50), 
                        (center_x, cart_y + cart_height // 2), 2)
        self.draw_text("CENTER", center_x - 40, cart_y + 60, self.GREEN, self.small_font)
        
        # Draw cart
        cart_rect = pygame.Rect(cart_x - cart_width // 2, cart_y - cart_height // 2, 
                               cart_width, cart_height)
        pygame.draw.rect(self.screen, self.BLUE, cart_rect)
        pygame.draw.rect(self.screen, self.WHITE, cart_rect, 2)
        
        # Draw wheels
        wheel_radius = 8
        pygame.draw.circle(self.screen, self.WHITE, 
                          (cart_x - cart_width // 3, cart_y + cart_height // 2), wheel_radius)
        pygame.draw.circle(self.screen, self.WHITE, 
                          (cart_x + cart_width // 3, cart_y + cart_height // 2), wheel_radius)
        
        # Draw pole
        pole_length = 100
        pole_end_x = cart_x + int(pole_length * np.sin(pole_angle))
        pole_end_y = cart_y - int(pole_length * np.cos(pole_angle))
        
        pygame.draw.line(self.screen, self.RED, (cart_x, cart_y), 
                        (pole_end_x, pole_end_y), 8)
        pygame.draw.circle(self.screen, self.YELLOW, (cart_x, cart_y), 10)
        pygame.draw.circle(self.screen, self.RED, (pole_end_x, pole_end_y), 12)
    
    def draw_stats(self):
        # Background panel - make taller for Q-values
        panel_height = 360 if (self.mode == "agent" and self.q_values is not None) else 280
        panel_rect = pygame.Rect(10, 10, 400, panel_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), panel_rect)
        pygame.draw.rect(self.screen, self.GREEN, panel_rect, 2)
        
        # Extract state
        cart_position = self.state[0]
        cart_velocity = self.state[1]
        pole_angle = self.state[2]
        pole_velocity = self.state[3]
        pole_angle_deg = np.degrees(pole_angle)
        
        # Mode info and angle limit
        mode_text = {
            "agent": "AI AGENT MODE (12° / max 1500 steps)",
            "manual": "MANUAL MODE - Pure Manual (60° / max 1500 steps)", 
            "both": "BATTLE MODE - You vs AI (60° / max 1500 steps)"
        }
        self.draw_text(mode_text.get(self.mode, ""), 20, 20, self.YELLOW, self.small_font)
        
        # Stats
        stats = [
            f"Steps: {self.steps}",
            f"Reward: {self.total_reward:.0f}",
            f"Cart Position: {cart_position:.3f} m",
            f"Cart Velocity: {cart_velocity:.3f} m/s",
            f"Pole Angle: {pole_angle_deg:.2f}°",
            f"Pole Velocity: {pole_velocity:.3f} rad/s",
        ]
        
        y_offset = 55
        for stat in stats:
            color = self.WHITE
            # Warn when approaching limit based on mode
            angle_warn_threshold = 10 if self.mode == "agent" else 50
            if "Pole Angle" in stat and abs(pole_angle_deg) > angle_warn_threshold:
                color = self.RED
            self.draw_text(stat, 20, y_offset, color, self.small_font)
            y_offset += 30
        
        # Add Q-values if in agent mode (separate section)
        if self.mode == "agent" and self.q_values is not None:
            y_offset += 10  # Extra spacing
            self.draw_text("Q-Values:", 20, y_offset, self.GREEN, self.small_font)
            y_offset += 30
            self.draw_text(f"  Left:  {self.q_values[0]:.2f}", 20, y_offset, self.WHITE, self.small_font)
            y_offset += 30
            self.draw_text(f"  Right: {self.q_values[1]:.2f}", 20, y_offset, self.WHITE, self.small_font)
            y_offset += 30
            action_text = '←LEFT' if self.q_values[0] > self.q_values[1] else 'RIGHT→'
            self.draw_text(f"Action: {action_text}", 20, y_offset, self.GREEN, self.small_font)
            y_offset += 35
        
        # Controls reminder at bottom of panel
        controls_y = panel_height - 35
        self.draw_text(":", 20, controls_y, self.GREEN, self.small_font)
        if self.mode == "manual":
            self.draw_text("← → : Move cart", 20, 270, self.WHITE, pygame.font.Font(None, 22))
        elif self.mode == "both":
            self.draw_text("← → : Push cart (AI tries to balance)", 20, 270, self.WHITE, pygame.font.Font(None, 22))
    
    def draw_footer(self):
        """Draw author credits and logo at bottom of screen during gameplay"""
        # Semi-transparent background bar
        footer_height = 50
        footer_surface = pygame.Surface((self.screen_width, footer_height))
        footer_surface.set_alpha(180)
        footer_surface.fill(self.BLACK)
        self.screen.blit(footer_surface, (0, self.screen_height - footer_height))
        
        # Author info on left
        self.draw_text("Davar Adil Yassin | Master's in AI - UKH | Advanced Robotics", 
                      20, self.screen_height - 35, (200, 200, 200), self.tiny_font)
        
        # Logo on right if available
        if self.logo_loaded:
            logo_small = pygame.transform.scale(self.ukh_logo, 
                                               (int(self.ukh_logo.get_width() * 0.4), 
                                                int(self.ukh_logo.get_height() * 0.4)))
            logo_x = self.screen_width - logo_small.get_width() - 20
            logo_y = self.screen_height - logo_small.get_height() - 10
            self.screen.blit(logo_small, (logo_x, logo_y))
    
    def draw_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        self.draw_text("GAME OVER!", self.screen_width // 2 - 150, self.screen_height // 2 - 100, 
                      self.RED, pygame.font.Font(None, 72))
        self.draw_text(f"Final Score: {self.total_reward:.0f}", self.screen_width // 2 - 150, 
                      self.screen_height // 2, self.WHITE, pygame.font.Font(None, 48))
        self.draw_text("Press R to restart or M for menu", self.screen_width // 2 - 250, 
                      self.screen_height // 2 + 80, self.YELLOW)
    
    def get_agent_action(self):
        """Get action from trained agent with Q-value info"""
        if not self.model_loaded:
            return self.env.action_space.sample()
        
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
            
            # Store Q-values for debugging (optional)
            self.q_values = q_values[0].numpy()
            
        return action
    
    def reset_game(self):
        self.create_environment(self.mode)
        self.total_reward = 0
        self.steps = 0
        self.game_over = False
        self.manual_action = None
        # Keep q_values if switching back to agent mode
        if self.mode != "agent":
            self.q_values = None
    
    def run(self):
        running = True
        
        while running:
            self.clock.tick(self.fps)
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    # Menu selection
                    if self.mode == "menu":
                        if event.key == pygame.K_1:
                            self.mode = "agent"
                            self.reset_game()
                        elif event.key == pygame.K_2:
                            self.mode = "manual"
                            self.reset_game()
                        elif event.key == pygame.K_3:
                            self.mode = "both"
                            self.reset_game()
                    
                    # Return to menu
                    if event.key == pygame.K_m:
                        self.mode = "menu"
                    
                    # Restart
                    if event.key == pygame.K_r and self.mode != "menu":
                        self.reset_game()
            
            # Handle continuous key presses for immediate responsive control
            if self.mode in ["manual", "both"] and not self.game_over:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.manual_action = 0  # Move left
                elif keys[pygame.K_RIGHT]:
                    self.manual_action = 1  # Move right
                else:
                    self.manual_action = None
            
            # Game logic
            if self.mode == "menu":
                self.draw_menu()
            
            elif self.mode in ["agent", "manual", "both"] and not self.game_over:
                # Determine action
                if self.mode == "agent":
                    action = self.get_agent_action()
                elif self.mode == "manual":
                    # Pure manual control - no AI assistance
                    if self.manual_action is not None:
                        action = self.manual_action
                        self.manual_action = None  # Clear after use
                    else:
                        # No action - let physics continue naturally
                        action = 1 if np.random.random() < 0.5 else 0  # Random to simulate no force
                elif self.mode == "both":
                    # Battle mode: You push, AI actively tries to counter-balance
                    if self.manual_action is not None:
                        # When you push, AI tries to balance by taking opposite or balancing action
                        user_action = self.manual_action
                        self.manual_action = None
                        # AI fights back by choosing its own action
                        ai_action = self.get_agent_action()
                        # Use user action (your push takes priority)
                        action = user_action
                    else:
                        # When you're not pushing, AI balances normally
                        action = self.get_agent_action()
                
                # Step environment
                self.state, reward, terminated, truncated, _ = self.env.step(action)
                self.total_reward += reward
                self.steps += 1
                
                if terminated or truncated:
                    self.game_over = True
                
                # Draw game
                self.screen.fill(self.BLACK)
                self.draw_cartpole()
                self.draw_stats()
                self.draw_footer()
            
            elif self.game_over:
                self.draw_cartpole()
                self.draw_stats()
                self.draw_footer()
                self.draw_game_over()
            
            pygame.display.flip()
        
        pygame.quit()
        self.env.close()
        sys.exit()

if __name__ == "__main__":
    game = InteractiveCartPole()
    game.run()
