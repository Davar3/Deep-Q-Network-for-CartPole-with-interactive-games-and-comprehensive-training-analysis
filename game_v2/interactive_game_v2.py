import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import numpy as np
import pygame # type: ignore
import sys
import os

"""
CartPole DQN Interactive Game V2 - Super Playable Edition
Author: Davar Adil Yassin
Master's Student in AI - University of Kurdistan Hewlêr (UKH)
Advanced Robotics Class Assignment
"""

# Super relaxed CartPole wrapper inspired by the web version
class SuperRelaxedCartPole:
    """CartPole with extremely forgiving physics for maximum playability"""
    def __init__(self, angle_limit=90.0, position_limit=5.0, max_steps=3000):
        self.env = gym.make("CartPole-v1")
        
        # Physics inspired by the web version - much more playable!
        self.env.unwrapped.gravity = 3.0  # Much much slower falling (web used 10 but we need even slower)
        self.env.unwrapped.tau = 0.01  # Smaller time step = smoother, slower simulation
        self.env.unwrapped.force_mag = 15.0  # Stronger pushes to counter the slower physics
        self.env.unwrapped.length = 1.0  # Longer pole = much easier to balance
        self.env.unwrapped.masspole = 0.05  # Lighter pole (default 0.1) - falls slower
        
        self.angle_limit = np.radians(angle_limit)
        self.position_limit = position_limit
        self.max_steps = max_steps
        self.current_step = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, **kwargs):
        self.current_step = 0
        result = self.env.reset(**kwargs)
        return result
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Super forgiving termination - only end if really extreme
        cart_position = state[0]
        pole_angle = state[2]
        
        terminated = (abs(pole_angle) > self.angle_limit or 
                     abs(cart_position) > self.position_limit)
        
        # Override 500 step limit
        if self.current_step >= self.max_steps:
            truncated = True
        else:
            truncated = False
        
        return state, reward, terminated, truncated, info
    
    def close(self):
        self.env.close()

# DQN Model
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

class SuperRelaxedCartPoleGame:
    def __init__(self):
        pygame.init()
        
        # Standard 16:9 aspect ratio screen
        self.screen_width = 1280
        self.screen_height = 720
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("CartPole V2 - Super Playable Edition!")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 100, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (160, 32, 240)
        
        # Fonts
        self.font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 30)
        self.tiny_font = pygame.font.Font(None, 24)
        
        # Game state
        self.mode = "menu"
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Environment
        self.env = None
        self.create_environment("agent")
        
        # Load UKH Logo
        try:
            logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Branding", "Logo-Transparent.png")
            if os.path.exists(logo_path):
                self.ukh_logo = pygame.image.load(logo_path)
                # Scale logo to reasonable size (height = 80px)
                logo_height = 80
                aspect_ratio = self.ukh_logo.get_width() / self.ukh_logo.get_height()
                logo_width = int(logo_height * aspect_ratio)
                self.ukh_logo = pygame.transform.scale(self.ukh_logo, (logo_width, logo_height))
                self.logo_loaded = True
            else:
                self.logo_loaded = False
        except:
            self.logo_loaded = False
        
        # Load model
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.model = DQN(input_dim, output_dim)
        
        # Use relative path to models folder
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dqn_cartpole_gymnasium.pth")
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
        self.high_score = 0
        self.q_values = None  # Store Q-values for debugging
    
    def create_environment(self, mode):
        if self.env is not None:
            self.env.close()
        
        if mode in ["manual", "both"]:
            # Super relaxed for manual play - 90 degrees, 3000 steps!
            self.env = SuperRelaxedCartPole(angle_limit=90.0, position_limit=5.0, max_steps=3000)
        else:
            # AI mode still gets some help but less forgiving
            self.env = SuperRelaxedCartPole(angle_limit=15.0, position_limit=2.4, max_steps=3000)
        
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
        
        # Title with gradient effect
        title = "CartPole V2: Super Playable Edition"
        self.draw_text(title, self.screen_width // 2 - 400, 80, self.ORANGE, pygame.font.Font(None, 72))
        
        # Display author info and logo
        if self.logo_loaded:
            logo_x = self.screen_width - self.ukh_logo.get_width() - 30
            logo_y = 20
            self.screen.blit(self.ukh_logo, (logo_x, logo_y))
        
        self.draw_text("By: Davar Adil Yassin", 50, 180, self.WHITE, self.small_font)
        self.draw_text("Master's Student in AI - UKH", 50, 210, self.WHITE, self.tiny_font)
        self.draw_text("Advanced Robotics Class", 50, 230, self.WHITE, self.tiny_font)
        
        self.draw_text("Select Mode:", self.screen_width // 2 - 120, 280, self.YELLOW)
        
        # Mode descriptions with better formatting
        modes = [
            ("1", "AI Agent Mode", "Watch the AI balance (15° limit, 3000 steps)", self.BLUE, 370),
            ("2", "Manual Mode", "You control everything (90° limit, 3000 steps)", self.GREEN, 460),
            ("3", "Battle Mode", "Fight the AI! You push, it tries to balance", self.PURPLE, 550),
        ]
        
        for key, title, desc, color, y in modes:
            self.draw_text(f"Press {key}:", 100, y, self.WHITE, self.small_font)
            self.draw_text(title, 250, y, color, self.small_font)
            self.draw_text(desc, 250, y + 30, self.WHITE, self.tiny_font)
        
        self.draw_text("Press ESC: Quit", 100, 620, self.RED, self.small_font)
        
        # High score
        if self.high_score > 0:
            self.draw_text(f"High Score: {self.high_score:.1f} seconds", 
                          self.screen_width // 2 - 200, 720, self.YELLOW, self.small_font)
        
        # Instructions
        instructions = [
            "Physics Improvements:",
            "• Slower gravity (30% less than standard)",
            "• Longer pole for easier balance",
            "• Gentler force application",
            "• Much more forgiving angles (90° vs 12°!)",
            "",
            "Controls:",
            "← → Arrow keys to control",
            "R to restart | M for menu"
        ]
        
        y_offset = 200
        x_offset = self.screen_width - 450
        for line in instructions:
            if line.startswith("•"):
                self.draw_text(line, x_offset + 20, y_offset, self.GREEN, self.tiny_font)
            else:
                self.draw_text(line, x_offset, y_offset, self.YELLOW if line.endswith(":") else self.WHITE, 
                             self.tiny_font if not line.endswith(":") else self.small_font)
            y_offset += 30
    
    def draw_cartpole(self):
        cart_position = self.state[0]
        pole_angle = self.state[2]
        
        # Drawing parameters
        scale = 120  # Slightly larger scale
        cart_x = self.screen_width // 2 + int(cart_position * scale)
        cart_y = self.screen_height // 2 + 150
        cart_width = 70
        cart_height = 45
        
        # Draw extended track
        track_y = cart_y + cart_height // 2
        pygame.draw.line(self.screen, self.WHITE, (50, track_y), 
                        (self.screen_width - 50, track_y), 4)
        
        # Draw center marker
        center_x = self.screen_width // 2
        pygame.draw.line(self.screen, self.GREEN, (center_x, track_y - 60), 
                        (center_x, track_y), 3)
        self.draw_text("CENTER", center_x - 45, cart_y + 70, self.GREEN, self.tiny_font)
        
        # Draw cart with shadow effect
        shadow_offset = 5
        shadow_rect = pygame.Rect(cart_x - cart_width // 2 + shadow_offset, 
                                  cart_y - cart_height // 2 + shadow_offset, 
                                  cart_width, cart_height - 2*8)
        pygame.draw.rect(self.screen, (50, 50, 50), shadow_rect, border_radius=5)
        
        cart_rect = pygame.Rect(cart_x - cart_width // 2, cart_y - cart_height // 2, 
                               cart_width, cart_height - 2*8)
        pygame.draw.rect(self.screen, self.BLUE, cart_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.WHITE, cart_rect, 3, border_radius=5)
        
        # Draw wheels
        wheel_radius = 10
        pygame.draw.circle(self.screen, (139, 69, 19), 
                          (cart_x - cart_width // 3, track_y), wheel_radius)
        pygame.draw.circle(self.screen, self.WHITE, 
                          (cart_x - cart_width // 3, track_y), wheel_radius, 2)
        pygame.draw.circle(self.screen, (139, 69, 19), 
                          (cart_x + cart_width // 3, track_y), wheel_radius)
        pygame.draw.circle(self.screen, self.WHITE, 
                          (cart_x + cart_width // 3, track_y), wheel_radius, 2)
        
        # Draw pole (thicker and more visible)
        pole_length = 140  # Longer for visibility
        pole_end_x = cart_x + int(pole_length * np.sin(pole_angle))
        pole_end_y = cart_y - int(pole_length * np.cos(pole_angle))
        
        pygame.draw.line(self.screen, self.RED, (cart_x, cart_y), 
                        (pole_end_x, pole_end_y), 10)
        pygame.draw.circle(self.screen, self.YELLOW, (cart_x, cart_y), 12)
        pygame.draw.circle(self.screen, self.ORANGE, (pole_end_x, pole_end_y), 15)
    
    def draw_stats(self):
        # Dynamic panel height based on Q-values display
        panel_height = 380 if (self.mode == "agent" and self.q_values is not None) else 320
        panel_rect = pygame.Rect(10, 10, 450, panel_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), panel_rect)
        pygame.draw.rect(self.screen, self.GREEN, panel_rect, 3)
        
        cart_position = self.state[0]
        cart_velocity = self.state[1]
        pole_angle = self.state[2]
        pole_velocity = self.state[3]
        pole_angle_deg = np.degrees(pole_angle)
        
        # Mode info
        mode_text = {
            "agent": "🤖 AI AGENT MODE (15° / 3000 steps)",
            "manual": "👤 MANUAL MODE (90° / 3000 steps)", 
            "both": "⚔️ BATTLE MODE (90° / 3000 steps)"
        }
        self.draw_text(mode_text.get(self.mode, ""), 20, 20, self.YELLOW, self.small_font)
        
        # Stats with color coding
        stats = [
            (f"Steps: {self.steps}", self.WHITE),
            (f"Score: {self.total_reward:.0f}", self.GREEN if self.total_reward > 100 else self.WHITE),
            (f"Time: {self.total_reward/10:.1f}s", self.WHITE),
            (f"Position: {cart_position:.2f}m", self.WHITE),
            (f"Velocity: {cart_velocity:.2f}m/s", self.WHITE),
            (f"Angle: {pole_angle_deg:.1f}°", 
             self.RED if abs(pole_angle_deg) > (10 if self.mode == "agent" else 70) else self.WHITE),
            (f"Angular Vel: {pole_velocity:.2f}rad/s", self.WHITE),
        ]
        
        y_offset = 70
        for stat_text, color in stats:
            self.draw_text(stat_text, 20, y_offset, color, self.small_font)
            y_offset += 38
        
        # Add Q-values if in agent mode
        if self.mode == "agent" and self.q_values is not None:
            y_offset += 10
            self.draw_text("Q-Values:", 20, y_offset, self.ORANGE, self.small_font)
            y_offset += 35
            self.draw_text(f"  Left:  {self.q_values[0]:.2f}", 20, y_offset, self.WHITE, self.small_font)
            y_offset += 35
            self.draw_text(f"  Right: {self.q_values[1]:.2f}", 20, y_offset, self.WHITE, self.small_font)
            y_offset += 35
            action_text = '←LEFT' if self.q_values[0] > self.q_values[1] else 'RIGHT→'
            self.draw_text(f"Action: {action_text}", 20, y_offset, self.GREEN, self.small_font)
            y_offset = panel_height - 60  # Adjust controls position
        
        # Controls
        controls_y = panel_height - 50 if (self.mode == "agent" and self.q_values is not None) else 280
        self.draw_text("Controls:", 20, controls_y, self.ORANGE, self.small_font)
        if self.mode == "manual":
            self.draw_text("← → Move cart", 20, controls_y + 30, self.WHITE, self.tiny_font)
        elif self.mode == "both":
            self.draw_text("← → Push (AI counters)", 20, controls_y + 30, self.WHITE, self.tiny_font)
    
    def draw_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        self.draw_text("GAME OVER!", self.screen_width // 2 - 200, self.screen_height // 2 - 120, 
                      self.RED, pygame.font.Font(None, 84))
        
        time_survived = self.total_reward / 10
        self.draw_text(f"You survived {time_survived:.1f} seconds!", 
                      self.screen_width // 2 - 250, self.screen_height // 2 - 20, 
                      self.WHITE, pygame.font.Font(None, 52))
        self.draw_text(f"Final Score: {self.total_reward:.0f}", 
                      self.screen_width // 2 - 200, self.screen_height // 2 + 40, 
                      self.YELLOW, pygame.font.Font(None, 52))
        
        if time_survived > self.high_score:
            self.high_score = time_survived
            self.draw_text("NEW HIGH SCORE! 🎉", self.screen_width // 2 - 200, 
                          self.screen_height // 2 + 100, self.GREEN, self.font)
        
        self.draw_text("Press R to restart or M for menu", self.screen_width // 2 - 280, 
                      self.screen_height // 2 + 160, self.YELLOW, self.small_font)
    
    def get_agent_action(self):
        """Get action from trained agent with Q-value info"""
        if not self.model_loaded:
            return self.env.action_space.sample()
        
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
            
            # Store Q-values for debugging
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
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
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
                    
                    if event.key == pygame.K_m:
                        self.mode = "menu"
                    
                    if event.key == pygame.K_r and self.mode != "menu":
                        self.reset_game()
            
            # Handle input
            if self.mode in ["manual", "both"] and not self.game_over:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.manual_action = 0
                elif keys[pygame.K_RIGHT]:
                    self.manual_action = 1
                else:
                    self.manual_action = None
            
            # Game logic
            if self.mode == "menu":
                self.draw_menu()
            
            elif self.mode in ["agent", "manual", "both"] and not self.game_over:
                if self.mode == "agent":
                    action = self.get_agent_action()
                elif self.mode == "manual":
                    if self.manual_action is not None:
                        action = self.manual_action
                        self.manual_action = None
                    else:
                        # No force when no input
                        action = 1 if np.random.random() < 0.5 else 0
                elif self.mode == "both":
                    if self.manual_action is not None:
                        action = self.manual_action
                        self.manual_action = None
                    else:
                        action = self.get_agent_action()
                
                self.state, reward, terminated, truncated, _ = self.env.step(action)
                self.total_reward += reward
                self.steps += 1
                
                if terminated or truncated:
                    self.game_over = True
                
                self.screen.fill(self.BLACK)
                self.draw_cartpole()
                self.draw_stats()
            
            elif self.game_over:
                self.draw_cartpole()
                self.draw_stats()
                self.draw_game_over()
            
            pygame.display.flip()
        
        pygame.quit()
        self.env.close()
        sys.exit()

if __name__ == "__main__":
    game = SuperRelaxedCartPoleGame()
    game.run()
