import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import heapq

# -------------------------------------------------------
# Simulation parameters
# -------------------------------------------------------
N_AGENTS = 12
WORLD_SIZE = 10.0
DT = 0.05
ROBOT_SPEED = 1.0
ROBOT_WIDTH = 0.30   # meters
ROBOT_LENGTH = 0.40  # meters
ROBOT_HALF_DIAG = np.sqrt(ROBOT_WIDTH**2 + ROBOT_LENGTH**2) / 2
ROBOT_RADIUS = ROBOT_HALF_DIAG   # ~0.25 m
AVOID_DIST = ROBOT_RADIUS * 4.0          # ~1.0 m
AVOID_GAIN = 2.5
GOAL_TOL = 0.5  # Increased tolerance to help with parking
WAYPOINT_TOL = 0.6
GRID_RES = 1.0
ROBOT_COLORS = [
    "blue", "red", "green", "purple", "orange", "brown", 
    "pink", "olive", "cyan", "magenta", "yellow", "darkblue"]

# Bicycle/Ackermann dynamics parameters
MAX_LINEAR_VEL = 1.2
MAX_REVERSE_VEL = 0.6  # Slower in reverse
MAX_LINEAR_ACCEL = 0.8
WHEELBASE = 0.35         # distance between front and rear axles (m)
MAX_STEER = np.radians(35)   # max steering angle (rad)
MAX_STEER_RATE = np.radians(45)  # max steering rate (rad/s)
REVERSE_THRESHOLD = np.radians(120)  # Start reversing if heading error > 120 degrees

# Replanning parameters
REPLAN_DIST = 2.0
REPLAN_COOLDOWN = 1.0
OBSTACLE_INFLATION = ROBOT_RADIUS * 2.0

goal_markers = []

# -------------------------------------------------------
# A* PATH PLANNING with Obstacle Grid
# -------------------------------------------------------
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def create_obstacle_grid(agents, current_agent_idx, grid_min, grid_max):
    """Create a grid with obstacles based on other robot positions"""
    grid_w = int((grid_max - grid_min) / GRID_RES)
    grid_h = int((grid_max - grid_min) / GRID_RES)
    obstacle_grid = np.zeros((grid_w, grid_h), dtype=bool)
    
    def to_grid(p):
        x = int((p[0] - grid_min) / GRID_RES)
        y = int((p[1] - grid_min) / GRID_RES)
        return x, y
    
    # Mark cells occupied by other robots
    for agent in agents:
        if agent.idx == current_agent_idx:
            continue
        
        gx, gy = to_grid(agent.position)
        
        # Inflate obstacle
        inflation_cells = int((OBSTACLE_INFLATION + ROBOT_HALF_DIAG) / GRID_RES)
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    if np.hypot(dx, dy) <= inflation_cells:
                        obstacle_grid[nx, ny] = True
    
    return obstacle_grid

def astar_with_obstacles(start, goal, obstacle_grid, grid_min, grid_max):
    """A* with obstacle avoidance"""
    
    def to_grid(p):
        return (int((p[0] - grid_min) / GRID_RES),
                int((p[1] - grid_min) / GRID_RES))
        
    def to_world(cell):
        return np.array([cell[0] * GRID_RES + grid_min + GRID_RES / 2,
                         cell[1] * GRID_RES + grid_min + GRID_RES / 2])

    start_c = to_grid(start)
    goal_c = to_grid(goal)
    
    grid_w, grid_h = obstacle_grid.shape

    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1)]

    open_set = []
    heapq.heappush(open_set, (0, start_c))
    came_from = {}
    g_score = {start_c: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_c:
            path = []
            while current in came_from:
                path.append(to_world(current))
                current = came_from[current]
            path.append(to_world(start_c))
            return path[::-1]

        for dx, dy in nbrs:
            nbr = (current[0] + dx, current[1] + dy)

            if not (0 <= nbr[0] < grid_w and 0 <= nbr[1] < grid_h):
                continue
            
            if obstacle_grid[nbr[0], nbr[1]]:
                continue

            tentative_g = g_score[current] + np.hypot(dx, dy)

            if nbr not in g_score or tentative_g < g_score[nbr]:
                g_score[nbr] = tentative_g
                f = tentative_g + heuristic(nbr, goal_c)
                heapq.heappush(open_set, (f, nbr))
                came_from[nbr] = current

    return [start, goal]


# -------------------------------------------------------
# Agent Class with Bicycle Kinematics
# -------------------------------------------------------
class Robot:
    def __init__(self, idx, color):
        self.idx = idx
        self.color = color
        self.position = np.random.uniform(-WORLD_SIZE, WORLD_SIZE, size=2)
        self.trail = [ self.position.copy() ]
        self.goal = np.random.uniform(-WORLD_SIZE, WORLD_SIZE, size=2)

        # Bicycle model state
        self.theta = np.random.uniform(0, 2*np.pi)  # heading angle
        self.linear_vel = 0.0  # forward velocity (can be negative for reverse)
        self.delta = 0.0  # steering angle
        self.is_reversing = False  # track if currently in reverse
        
        self.reached = False
        
        # Initial path
        obstacle_grid = np.zeros((int(2*WORLD_SIZE/GRID_RES), 
                                 int(2*WORLD_SIZE/GRID_RES)), dtype=bool)
        self.path = astar_with_obstacles(
            self.position, self.goal, obstacle_grid,
            grid_min=-WORLD_SIZE, grid_max=+WORLD_SIZE
        )
        self.path_i = 0
        
        # Replanning tracking
        self.time_since_replan = 0.0
        self.replan_count = 0
        self.is_replanning = False

    def detect_obstacles_ahead(self, agents):
        """Check if there are obstacles blocking the path ahead"""
        if self.path_i >= len(self.path):
            return False
        
        lookahead = min(3, len(self.path) - self.path_i)
        
        for i in range(lookahead):
            wp_idx = self.path_i + i
            if wp_idx >= len(self.path):
                break
            
            waypoint = self.path[wp_idx]
            
            for other in agents:
                if other.idx == self.idx:
                    continue
                
                dist_to_waypoint = np.linalg.norm(other.position - waypoint)
                if dist_to_waypoint < REPLAN_DIST:
                    return True
        
        return False

    def replan_path(self, agents):
        """Recompute A* path avoiding current obstacle positions"""
        self.is_replanning = True
        
        obstacle_grid = create_obstacle_grid(
            agents, self.idx, 
            grid_min=-WORLD_SIZE, grid_max=+WORLD_SIZE
        )
        
        new_path = astar_with_obstacles(
            self.position, self.goal, obstacle_grid,
            grid_min=-WORLD_SIZE, grid_max=+WORLD_SIZE
        )
        
        if len(new_path) > 1:
            self.path = new_path
            self.path_i = 0
            self.replan_count += 1
        
        self.time_since_replan = 0.0
        self.is_replanning = False

    def get_waypoint(self):
        if self.path_i >= len(self.path):
            return self.goal
        return self.path[self.path_i]

    def advance_waypoint(self):
        if np.linalg.norm(self.get_waypoint() - self.position) < WAYPOINT_TOL:
            self.path_i += 1

    def compute_desired_direction(self, agents):
        """Compute desired heading considering waypoint and obstacle avoidance"""
        wp = self.get_waypoint()
        
        to_waypoint = wp - self.position
        dist_to_wp = np.linalg.norm(to_waypoint)
        
        if dist_to_wp < GOAL_TOL:
            return None, 0.0
        
        desired_dir = to_waypoint / dist_to_wp
        
        # Add repulsive components from nearby robots
        for other in agents:
            if other.idx == self.idx:
                continue
            diff = self.position - other.position
            center_dist = np.linalg.norm(diff)
            min_safe = ROBOT_HALF_DIAG * 2.0

            if center_dist < min_safe + AVOID_DIST:
                overlap = (min_safe + AVOID_DIST) - center_dist
                repulse = (diff / center_dist) * (AVOID_GAIN * overlap)
                desired_dir += repulse
        
        norm = np.linalg.norm(desired_dir)
        if norm > 1e-6:
            desired_dir = desired_dir / norm
        
        desired_angle = np.arctan2(desired_dir[1], desired_dir[0])
        
        # Reduce speed dramatically when close to goal to help with parking
        if dist_to_wp < 1.5:
            desired_speed = max(0.3, dist_to_wp * 0.4)  # Minimum speed to keep moving
        else:
            desired_speed = min(ROBOT_SPEED, dist_to_wp * 2)
        
        return desired_angle, desired_speed

    def bicycle_control(self, desired_angle, desired_speed):
        """
        Pure pursuit style control for bicycle model with reverse capability.
        Returns: (target_velocity, target_steering_angle)
        """
        # Compute heading error
        heading_error = np.arctan2(np.sin(desired_angle - self.theta),
                                    np.cos(desired_angle - self.theta))
        
        # Decide whether to go forward or reverse
        should_reverse = abs(heading_error) > REVERSE_THRESHOLD
        
        if should_reverse:
            # Going in reverse - flip the heading error and use negative velocity
            self.is_reversing = True
            # When reversing, we want to steer away from the goal to turn around
            reverse_heading_error = np.arctan2(np.sin(desired_angle - self.theta + np.pi),
                                               np.cos(desired_angle - self.theta + np.pi))
            
            kp_steer = 2.0  # Gentler steering in reverse
            target_delta = np.clip(kp_steer * reverse_heading_error, -MAX_STEER, MAX_STEER)
            
            # Slower speed in reverse
            target_v = -min(desired_speed * 0.5, MAX_REVERSE_VEL)
            
        else:
            # Normal forward driving
            self.is_reversing = False
            
            # Proportional steering control
            kp_steer = 2.5
            target_delta = np.clip(kp_steer * heading_error, -MAX_STEER, MAX_STEER)
            
            # Reduce speed when making sharp turns
            turn_factor = abs(target_delta) / MAX_STEER
            speed_reduction = 1.0 - 0.6 * turn_factor
            target_v = min(desired_speed, MAX_LINEAR_VEL) * speed_reduction
        
        return target_v, target_delta

    def apply_control_limits(self, target_vel, target_delta):
        """Apply velocity and steering rate limits"""
        # Limit linear acceleration (works for both forward and reverse)
        vel_change = target_vel - self.linear_vel
        max_vel_change = MAX_LINEAR_ACCEL * DT
        vel_change = np.clip(vel_change, -max_vel_change, max_vel_change)
        self.linear_vel += vel_change
        self.linear_vel = np.clip(self.linear_vel, -MAX_REVERSE_VEL, MAX_LINEAR_VEL)
        
        # Limit steering rate
        delta_change = target_delta - self.delta
        max_delta_change = MAX_STEER_RATE * DT
        delta_change = np.clip(delta_change, -max_delta_change, max_delta_change)
        self.delta += delta_change
        self.delta = np.clip(self.delta, -MAX_STEER, MAX_STEER)

    def update_bicycle_kinematics(self):
        """
        Update position and orientation using bicycle/Ackermann model:
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = (v / L) * tan(delta)
        
        where L is the wheelbase and delta is the steering angle
        """
        if abs(self.linear_vel) < 1e-6:
            return
        
        # Bicycle model kinematics
        self.theta += (self.linear_vel / WHEELBASE) * np.tan(self.delta) * DT
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))  # Normalize
        
        self.position[0] += self.linear_vel * np.cos(self.theta) * DT
        self.position[1] += self.linear_vel * np.sin(self.theta) * DT

    def update(self, agents):
        # Check if reached goal first
        if np.linalg.norm(self.position - self.goal) < GOAL_TOL:
            if not self.reached:
                # Just reached - stop all motion immediately
                self.reached = True
                self.linear_vel = 0.0
                self.delta = 0.0
            return
        
        # Update replan timer
        self.time_since_replan += DT
        
        # Check if replanning is needed
        if (self.time_since_replan > REPLAN_COOLDOWN and 
            self.detect_obstacles_ahead(agents)):
            self.replan_path(agents)
        
        self.advance_waypoint()
        
        desired_angle, desired_speed = self.compute_desired_direction(agents)
        
        # Compute bicycle controls
        target_vel, target_delta = self.bicycle_control(desired_angle, desired_speed)
        
        # Apply limits
        self.apply_control_limits(target_vel, target_delta)
        
        # Update kinematics
        self.update_bicycle_kinematics()


# -------------------------------------------------------
# Create agents
# -------------------------------------------------------
agents = [Robot(i, color=ROBOT_COLORS[i % len(ROBOT_COLORS)]) 
          for i in range(N_AGENTS)]

# -------------------------------------------------------
# Visualization Setup
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))
ax.set_xlim(-WORLD_SIZE, WORLD_SIZE)
ax.set_ylim(-WORLD_SIZE, WORLD_SIZE)
ax.set_aspect("equal")
ax.set_title("Multi-Robot A* with Bicycle/Ackermann Kinematics", fontsize=14, fontweight='bold')

status_text = ax.text(
    0.02, 0.98, "",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

points, = ax.plot([], [], "bo", markersize=10, label="Robots")
goals, = ax.plot([], [], "rx", markersize=10, markeredgewidth=2, label="Goals")
paths_plots = [
    ax.plot([], [], linestyle="--", linewidth=1.5, alpha=0.4, 
            color=agents[i].color)[0]
    for i in range(N_AGENTS)
]
trail_plots = [
    ax.plot([], [], linestyle="-", linewidth=1.8, alpha=0.8,
            color=agents[i].color)[0]
    for i in range(N_AGENTS)
]

replan_plots = [
    ax.plot(
        [], [], 
        linestyle="-", 
        linewidth=3, 
        alpha=0.9,
        color=agents[i].color
    )[0]
    for i in range(N_AGENTS)
]

from matplotlib.patches import Circle
arrows = []
detection_circles = []

ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

def init():
    status_text.set_text("")
    points.set_data([], [])

    goal_markers.clear()

    for agent in agents:
        gm, = ax.plot(
            agent.goal[0], agent.goal[1],
            marker='x',
            markersize=12,
            markeredgewidth=2,
            linestyle='None',
            color=agent.color
        )
        goal_markers.append(gm)

    return [points, goals, status_text] + goal_markers + paths_plots + replan_plots + trail_plots

rectangles = []

def update(frame):
    global rectangles
    for r in rectangles:
        r.remove()
    rectangles.clear()

    for agent in agents:
        agent.update(agents)
        agent.trail.append(agent.position.copy())
    
    reached_count = sum(a.reached for a in agents)
    total_replans = sum(a.replan_count for a in agents)
    
    status_text.set_text(
        f"Reached: {reached_count}/{N_AGENTS}\n"
        f"Total Replans: {total_replans}"
    )
    
    xs = [a.position[0] for a in agents]
    ys = [a.position[1] for a in agents]
    
    gx = [a.goal[0] for a in agents]
    gy = [a.goal[1] for a in agents]
    
    for tp, agent in zip(trail_plots, agents):
        pts = np.array(agent.trail)
        tp.set_data(pts[:, 0], pts[:, 1])
    
    for arrow in arrows:
        arrow.remove()
    for circle in detection_circles:
        circle.remove()
    arrows.clear()
    detection_circles.clear()
    
    # Draw rectangles (robots) with color coding for reverse
    for agent in agents:
        if agent.reached:
            color = agent.color
        elif agent.is_replanning:
            color = "yellow"
        elif agent.is_reversing:
            color = "orange"  # Orange when reversing
        else:
            color = agent.color

        rect = Rectangle(
            (agent.position[0] - ROBOT_LENGTH/2, agent.position[1] - ROBOT_WIDTH/2),
            ROBOT_LENGTH,
            ROBOT_WIDTH,
            angle=np.degrees(agent.theta),
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(rect)
        rectangles.append(rect)

        if not agent.reached:
            circle = ax.add_patch(Circle(
                agent.position, REPLAN_DIST,
                fill=False, edgecolor='red', linewidth=0.5,
                linestyle='--', alpha=0.2
            ))
            detection_circles.append(circle)
    
    for pl, agent in zip(paths_plots, agents):
        if not agent.is_replanning:
            pts = np.array(agent.path)
            pl.set_data(pts[:, 0], pts[:, 1])
        else:
            pl.set_data([], [])
    
    for rpl, agent in zip(replan_plots, agents):
        if agent.time_since_replan < 0.5:
            pts = np.array(agent.path)
            rpl.set_data(pts[:, 0], pts[:, 1])
        else:
            rpl.set_data([], [])
    
    return (
        [points, status_text]
        + goal_markers
        + paths_plots
        + replan_plots
        + trail_plots
        + rectangles
        + detection_circles
    )

anim = FuncAnimation(fig, update, init_func=init, frames=3000, interval=30, blit=True)
plt.show()