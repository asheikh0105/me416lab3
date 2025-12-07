import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# === CONFIGURATION ===

@dataclass
class Config:
    """Controller and Planning Configuration Parameters."""
    # Pure Pursuit Controller
    lookaheadDist: float = 0.4
    targetLinearVel: float = 0.3
    maxLinVel: float = 0.3
    maxAngVel: float = np.deg2rad(90)
    
    # Robot Dimensions (car-like)
    robotLength: float = 0.40  # Wheelbase
    robotWidth: float = 0.30
    safetyBuffer: float = 0.15
    
    # Dubins Parameters
    maxSteeringAngle: float = np.deg2rad(35)
    
    # Path planning - SWITCHED TO A* for speed
    replanInterval: float = 0.5  # More frequent replanning
    gridResolution: float = 0.15  # Grid cell size for A*
    
    # Dynamic obstacle parameters
    predictionHorizon: float = 2.0
    
    # Local Collision Avoidance
    coordinationHorizon: float = 1.5
    stopDistance: float = 0.6
    emergencyDistance: float = 0.4
    
    # Tolerances
    posTolerance: float = 0.25
    headingTolerance: float = np.deg2rad(30)
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.minTurnRadius = self.robotLength / np.tan(self.maxSteeringAngle)
        # Clearance radius: diagonal of box + safety buffer
        R_diag = np.sqrt(self.robotWidth**2 + self.robotLength**2) / 2.0
        self.robotClearanceRadius = R_diag + self.safetyBuffer
        self.safeSeparation = 2 * self.robotClearanceRadius

class SimParams:
    """Simulation Parameters."""
    def __init__(self):
        self.numRobots: int = 12
        self.maxTime: float = 300.0
        self.dt: float = 1.0 / 30.0
        self.mapSize: np.ndarray = np.array([5.5, 5.0])
        self.realTimeMode: bool = True

# === DATA STRUCTURES ===

class Path:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self.s = self._calculate_arc_length(x, y)
    
    @staticmethod
    def _calculate_arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        return np.insert(np.cumsum(ds), 0, 0.0)

class RobotState:
    def __init__(self, initial_pose: np.ndarray, goal: np.ndarray, initial_path: Path):
        self.pose = initial_pose.copy()
        self.velocity = np.array([0.0, 0.0])
        self.goal = goal
        self.trajectory = [initial_pose[:2].copy()]
        self.reached = False
        self.reachedTime = np.inf
        self.lastReplanTime = -np.inf
        self.replanCount = 0
        self.path = initial_path
        self.stuckCounter = 0
        self.lastPosition = initial_pose[:2].copy()

# === HELPER FUNCTIONS ===

def wrap_to_pi(angle: float) -> float:
    """Wraps an angle to the range (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def get_oriented_box_points(x: float, y: float, theta: float, length: float, width: float, num_points: int = 8) -> np.ndarray:
    """Get points around the perimeter of an oriented bounding box."""
    half_l, half_w = length / 2, width / 2
    
    # Create points around perimeter
    perimeter_points = []
    # Front edge
    for i in range(num_points // 4):
        t = i / (num_points // 4 - 1) if num_points > 4 else 0
        perimeter_points.append([half_l, -half_w + t * 2 * half_w])
    # Right edge
    for i in range(num_points // 4):
        t = i / (num_points // 4 - 1) if num_points > 4 else 0
        perimeter_points.append([half_l - t * 2 * half_l, half_w])
    # Back edge
    for i in range(num_points // 4):
        t = i / (num_points // 4 - 1) if num_points > 4 else 0
        perimeter_points.append([-half_l, half_w - t * 2 * half_w])
    # Left edge
    for i in range(num_points // 4):
        t = i / (num_points // 4 - 1) if num_points > 4 else 0
        perimeter_points.append([-half_l + t * 2 * half_l, -half_w])
    
    local_points = np.array(perimeter_points).T
    
    # Rotation and translation
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    global_points = R @ local_points
    global_points[0, :] += x
    global_points[1, :] += y
    
    return global_points.T

def check_box_collision(pose1: np.ndarray, pose2: np.ndarray, cfg: Config) -> bool:
    """Check if two oriented bounding boxes collide using separating axis theorem."""
    x1, y1, theta1 = pose1
    x2, y2, theta2 = pose2
    
    # Quick distance check first
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if dist > (cfg.robotLength + cfg.robotWidth):  # Conservative bound
        return False
    
    # Get corner points
    def get_corners(x, y, theta):
        hl, hw = cfg.robotLength / 2, cfg.robotWidth / 2
        corners = np.array([
            [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]
        ])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = (R @ corners.T).T
        return rotated + np.array([x, y])
    
    corners1 = get_corners(x1, y1, theta1)
    corners2 = get_corners(x2, y2, theta2)
    
    # Separating Axis Theorem - test box1 axes and box2 axes
    axes = [
        np.array([np.cos(theta1), np.sin(theta1)]),
        np.array([-np.sin(theta1), np.cos(theta1)]),
        np.array([np.cos(theta2), np.sin(theta2)]),
        np.array([-np.sin(theta2), np.cos(theta2)])
    ]
    
    for axis in axes:
        # Project all corners onto axis
        proj1 = corners1 @ axis
        proj2 = corners2 @ axis
        
        # Check for overlap
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False  # Found separating axis - no collision
    
    return True  # No separating axis found - collision

# === A* PATH PLANNER (Much faster than RRT*) ===

class AStarPlanner:
    """Grid-based A* planner with dynamic obstacle handling."""
    
    def __init__(self, map_size: np.ndarray, resolution: float, cfg: Config):
        self.map_size = map_size
        self.resolution = resolution
        self.cfg = cfg
        
        self.grid_width = int(np.ceil(map_size[0] / resolution))
        self.grid_height = int(np.ceil(map_size[1] / resolution))
        
        # 8-connectivity
        self.motions = [
            [1, 0, 1.0], [0, 1, 1.0], [-1, 0, 1.0], [0, -1, 1.0],
            [1, 1, 1.414], [1, -1, 1.414], [-1, 1, 1.414], [-1, -1, 1.414]
        ]
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int(np.floor(x / self.resolution))
        gy = int(np.floor(y / self.resolution))
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = (gx + 0.5) * self.resolution
        y = (gy + 0.5) * self.resolution
        return x, y
    
    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid cell is within bounds."""
        return 0 <= gx < self.grid_width and 0 <= gy < self.grid_height
    
    def create_obstacle_map(self, other_robots: List[Dict]) -> np.ndarray:
        """Create binary obstacle map from other robots."""
        obstacle_map = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        
        inflation_cells = int(np.ceil(self.cfg.robotClearanceRadius / self.resolution))
        
        for robot in other_robots:
            # Current position
            cx, cy = self.world_to_grid(robot['x'], robot['y'])
            if self.is_valid(cx, cy):
                for dx in range(-inflation_cells, inflation_cells + 1):
                    for dy in range(-inflation_cells, inflation_cells + 1):
                        gx, gy = cx + dx, cy + dy
                        if self.is_valid(gx, gy):
                            wx, wy = self.grid_to_world(gx, gy)
                            dist = np.sqrt((wx - robot['x'])**2 + (wy - robot['y'])**2)
                            if dist < self.cfg.robotClearanceRadius:
                                obstacle_map[gx, gy] = True
            
            # Predicted positions
            vx, vy = robot['vx'], robot['vy']
            speed = np.sqrt(vx**2 + vy**2)
            
            if speed > 0.05:
                pred_dist = speed * self.cfg.predictionHorizon
                num_preds = max(3, int(pred_dist / self.resolution))
                
                for i in range(1, num_preds + 1):
                    t = (i / num_preds) * self.cfg.predictionHorizon
                    px = robot['x'] + vx * t
                    py = robot['y'] + vy * t
                    
                    pgx, pgy = self.world_to_grid(px, py)
                    if self.is_valid(pgx, pgy):
                        pred_inflation = max(1, int(inflation_cells * 0.7))
                        for dx in range(-pred_inflation, pred_inflation + 1):
                            for dy in range(-pred_inflation, pred_inflation + 1):
                                gx, gy = pgx + dx, pgy + dy
                                if self.is_valid(gx, gy):
                                    obstacle_map[gx, gy] = True
        
        return obstacle_map
    
    def plan(self, start: np.ndarray, goal: np.ndarray, obstacle_map: np.ndarray) -> Optional[Path]:
        """A* path planning."""
        start_gx, start_gy = self.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.world_to_grid(goal[0], goal[1])
        
        if not (self.is_valid(start_gx, start_gy) and self.is_valid(goal_gx, goal_gy)):
            return None
        
        # Initialize
        open_set = {(start_gx, start_gy)}
        closed_set = set()
        
        g_score = {(start_gx, start_gy): 0}
        f_score = {(start_gx, start_gy): self._heuristic(start_gx, start_gy, goal_gx, goal_gy)}
        came_from = {}
        
        while open_set:
            # Get node with lowest f_score
            current = min(open_set, key=lambda n: f_score.get(n, np.inf))
            
            if current == (goal_gx, goal_gy):
                # Reconstruct path
                path_grid = [current]
                while current in came_from:
                    current = came_from[current]
                    path_grid.append(current)
                path_grid.reverse()
                
                # Convert to world coordinates
                path_x = []
                path_y = []
                for gx, gy in path_grid:
                    wx, wy = self.grid_to_world(gx, gy)
                    path_x.append(wx)
                    path_y.append(wy)
                
                return Path(np.array(path_x), np.array(path_y))
            
            open_set.remove(current)
            closed_set.add(current)
            
            # Explore neighbors
            for motion in self.motions:
                neighbor = (current[0] + motion[0], current[1] + motion[1])
                
                if not self.is_valid(neighbor[0], neighbor[1]):
                    continue
                
                if obstacle_map[neighbor[0], neighbor[1]]:
                    continue
                
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + motion[2]
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g >= g_score.get(neighbor, np.inf):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + self._heuristic(neighbor[0], neighbor[1], goal_gx, goal_gy)
        
        return None  # No path found
    
    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# === PURE PURSUIT CONTROLLER ===

def find_lookahead_point(robot_x: float, robot_y: float, path: Path, lookahead_dist: float) -> Tuple[float, float, float]:
    """Find lookahead point on path."""
    distances = np.sqrt((path.x - robot_x)**2 + (path.y - robot_y)**2)
    closest_idx = np.argmin(distances)
    cross_track_error = distances[closest_idx]
    
    # Target arc length
    s_closest = path.s[closest_idx]
    s_target = s_closest + lookahead_dist
    
    if s_target >= path.s[-1]:
        return path.x[-1], path.y[-1], cross_track_error
    
    idx = np.searchsorted(path.s, s_target)
    if idx >= len(path.x):
        idx = len(path.x) - 1
    
    return path.x[idx], path.y[idx], cross_track_error

def pure_pursuit_control(robot: RobotState, cfg: Config) -> Tuple[float, float]:
    """Pure Pursuit controller with Dubins constraints."""
    lookahead_x, lookahead_y, _ = find_lookahead_point(
        robot.pose[0], robot.pose[1], robot.path, cfg.lookaheadDist
    )
    
    # Calculate desired heading
    dx = lookahead_x - robot.pose[0]
    dy = lookahead_y - robot.pose[1]
    desired_heading = np.arctan2(dy, dx)
    heading_error = wrap_to_pi(desired_heading - robot.pose[2])
    
    # Linear velocity (forward only)
    if np.abs(heading_error) < np.deg2rad(90):
        lin_vel = cfg.targetLinearVel * np.cos(heading_error)
    else:
        lin_vel = cfg.targetLinearVel * 0.1  # Move slowly when target is behind
    
    # Angular velocity
    dist_to_lookahead = np.sqrt(dx**2 + dy**2)
    if dist_to_lookahead > 1e-3:
        ang_vel = (2 * lin_vel * np.sin(heading_error)) / dist_to_lookahead
    else:
        ang_vel = 0.0
    
    # Apply Dubins constraints
    if lin_vel > 0.01:
        max_ang_vel = lin_vel / cfg.minTurnRadius
    else:
        max_ang_vel = cfg.maxAngVel
    
    ang_vel = np.clip(ang_vel, -max_ang_vel, max_ang_vel)
    lin_vel = np.clip(lin_vel, 0.0, cfg.maxLinVel)
    
    return lin_vel, ang_vel

# === LOCAL COLLISION AVOIDANCE ===

def negotiate_velocity(my_robot: RobotState, all_robots: Dict[str, RobotState], my_id: str, cfg: Config, v_ideal: float) -> float:
    """Priority-based velocity negotiation."""
    my_pose = my_robot.pose
    my_dist_to_goal = np.sqrt((my_pose[0] - my_robot.goal[0])**2 + (my_pose[1] - my_robot.goal[1])**2)
    
    v_cmd = v_ideal
    
    for rid, other_robot in all_robots.items():
        if rid == my_id or other_robot.reached:
            continue
        
        other_pose = other_robot.pose
        
        # Current distance
        curr_dist = np.sqrt((my_pose[0] - other_pose[0])**2 + (my_pose[1] - other_pose[1])**2)
        
        # Emergency stop
        if curr_dist < cfg.emergencyDistance:
            return 0.0
        
        # Check for box collision at current positions
        if check_box_collision(my_pose, other_pose, cfg):
            return 0.0
        
        # Predict future positions
        T = cfg.coordinationHorizon
        my_pred_x = my_pose[0] + v_ideal * np.cos(my_pose[2]) * T
        my_pred_y = my_pose[1] + v_ideal * np.sin(my_pose[2]) * T
        my_pred_pose = np.array([my_pred_x, my_pred_y, my_pose[2]])
        
        other_vx, other_vy = other_robot.velocity
        other_pred_x = other_pose[0] + other_vx * T
        other_pred_y = other_pose[1] + other_vy * T
        other_pred_pose = np.array([other_pred_x, other_pred_y, other_pose[2]])
        
        # Check predicted collision
        pred_dist = np.sqrt((my_pred_x - other_pred_x)**2 + (my_pred_y - other_pred_y)**2)
        
        if pred_dist < cfg.safeSeparation or check_box_collision(my_pred_pose, other_pred_pose, cfg):
            # Priority based on distance to goal
            other_dist_to_goal = np.sqrt((other_pose[0] - other_robot.goal[0])**2 + (other_pose[1] - other_robot.goal[1])**2)
            
            if my_dist_to_goal > other_dist_to_goal:
                # Other robot has priority
                v_cmd = min(v_cmd, v_ideal * 0.3)
    
    return max(0.0, v_cmd)

# === MAIN SIMULATOR ===

def multi_robot_simulator():
    """Main simulation loop."""
    np.random.seed(42)
    
    SIM = SimParams()
    CFG = Config()
    
    print(f'Configuration:')
    print(f'  Min Turn Radius: {CFG.minTurnRadius:.3f}m')
    print(f'  Robot Size: {CFG.robotLength:.2f}m x {CFG.robotWidth:.2f}m')
    print(f'  Clearance Radius: {CFG.robotClearanceRadius:.3f}m')
    print(f'  Grid Resolution: {CFG.gridResolution:.3f}m')
    
    # Initialize A* planner
    planner = AStarPlanner(SIM.mapSize, CFG.gridResolution, CFG)
    
    # Initialize robots
    print(f'\nInitializing {SIM.numRobots} robots...')
    robots: Dict[str, RobotState] = {}
    min_separation = 2 * CFG.robotClearanceRadius + 0.2
    edge_margin = 0.5
    
    for i in range(1, SIM.numRobots + 1):
        robot_id = f'robot{i:02d}'
        
        # Find valid start
        for _ in range(100):
            x = edge_margin + np.random.rand() * (SIM.mapSize[0] - 2 * edge_margin)
            y = edge_margin + np.random.rand() * (SIM.mapSize[1] - 2 * edge_margin)
            
            valid = True
            for robot in robots.values():
                if np.sqrt((x - robot.pose[0])**2 + (y - robot.pose[1])**2) < min_separation:
                    valid = False
                    break
            if valid:
                break
        
        # Find valid goal
        for _ in range(100):
            gx = edge_margin + np.random.rand() * (SIM.mapSize[0] - 2 * edge_margin)
            gy = edge_margin + np.random.rand() * (SIM.mapSize[1] - 2 * edge_margin)
            
            if np.sqrt((gx - x)**2 + (gy - y)**2) > 2.0:
                break
        
        theta = np.random.rand() * 2 * np.pi
        initial_pose = np.array([x, y, theta])
        goal_pos = np.array([gx, gy])
        
        # Initial straight-line path
        path_x = np.linspace(x, gx, 20)
        path_y = np.linspace(y, gy, 20)
        initial_path = Path(path_x, path_y)
        
        robots[robot_id] = RobotState(initial_pose, goal_pos, initial_path)
        print(f'  {robot_id}: Start({x:.2f}, {y:.2f}) -> Goal({gx:.2f}, {gy:.2f})')
    
    # Visualization setup
    if SIM.realTimeMode:
        plt.ion()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim([-0.5, SIM.mapSize[0]])
    ax.set_ylim([-0.5, SIM.mapSize[1]])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    colors = plt.cm.get_cmap('hsv', SIM.numRobots)
    plot_handles = {}
    
    for i, (rid, robot) in enumerate(robots.items()):
        col = colors(i)
        plot_handles[rid] = {
            'goal': ax.plot(robot.goal[0], robot.goal[1], 'x', color=col, markersize=12, markeredgewidth=3)[0],
            'path': ax.plot([], [], '--', color=col, linewidth=1, alpha=0.4)[0],
            'traj': ax.plot([], [], '-', color=col, linewidth=1.5, alpha=0.7)[0],
            'robot': ax.plot([], [], 'o', color=col, markersize=10)[0],
            'heading': ax.plot([], [], '-', color=col, linewidth=2)[0],
            'box': ax.plot([], [], '-', color=col, linewidth=1.5)[0]
        }
    
    title = ax.set_title('Multi-Robot Simulation (A* + Dubins)')
    
    # Simulation loop
    print(f'\nStarting simulation...\n')
    sim_time = 0.0
    frame_count = 0
    
    while sim_time < SIM.maxTime:
        frame_start = time.time()
        
        # Update velocities for all robots
        for robot in robots.values():
            if not robot.reached:
                robot.velocity = np.array([
                    CFG.targetLinearVel * np.cos(robot.pose[2]),
                    CFG.targetLinearVel * np.sin(robot.pose[2])
                ])
        
        # Control each robot
        num_reached = sum(1 for r in robots.values() if r.reached)
        
        for rid, robot in robots.items():
            if robot.reached:
                continue
            
            # Check goal
            dist_to_goal = np.sqrt((robot.pose[0] - robot.goal[0])**2 + (robot.pose[1] - robot.goal[1])**2)
            if dist_to_goal < CFG.posTolerance:
                robot.reached = True
                robot.reachedTime = sim_time
                print(f'  [{sim_time:.1f}s] {rid} reached goal ({robot.replanCount} replans)')
                continue
            
            # Check if stuck
            pos_change = np.linalg.norm(robot.pose[:2] - robot.lastPosition)
            if pos_change < 0.01:
                robot.stuckCounter += 1
            else:
                robot.stuckCounter = 0
            robot.lastPosition = robot.pose[:2].copy()
            
            # Replan if needed
            should_replan = (
                (sim_time - robot.lastReplanTime) >= CFG.replanInterval or
                robot.stuckCounter > 15
            )
            
            if should_replan:
                # Create obstacle map from other robots
                other_robots_list = []
                for other_id, other_robot in robots.items():
                    if other_id != rid and not other_robot.reached:
                        other_robots_list.append({
                            'x': other_robot.pose[0],
                            'y': other_robot.pose[1],
                            'vx': other_robot.velocity[0],
                            'vy': other_robot.velocity[1]
                        })
                
                obstacle_map = planner.create_obstacle_map(other_robots_list)
                new_path = planner.plan(robot.pose, robot.goal, obstacle_map)
                
                if new_path is not None:
                    robot.path = new_path
                    robot.replanCount += 1
                    robot.stuckCounter = 0
                    plot_handles[rid]['path'].set_data(robot.path.x, robot.path.y)
                
                robot.lastReplanTime = sim_time
            
            # Pure pursuit control
            lin_vel, ang_vel = pure_pursuit_control(robot, CFG)
            
            # Collision avoidance
            lin_vel = negotiate_velocity(robot, robots, rid, CFG, lin_vel)
            
            # Update state
            robot.pose[0] += lin_vel * np.cos(robot.pose[2]) * SIM.dt
            robot.pose[1] += lin_vel * np.sin(robot.pose[2]) * SIM.dt
            robot.pose[2] = wrap_to_pi(robot.pose[2] + ang_vel * SIM.dt)
            
            robot.trajectory.append(robot.pose[:2].copy())
        
        # Update visualization
        if frame_count % 2 == 0 or not SIM.realTimeMode:
            for rid, robot in robots.items():
                if robot.reached:
                    plot_handles[rid]['robot'].set_data([], [])
                    plot_handles[rid]['heading'].set_data([], [])
                    plot_handles[rid]['box'].set_data([], [])
                    continue
                
                # Robot position
                plot_handles[rid]['robot'].set_data([robot.pose[0]], [robot.pose[1]])
                
                # Heading arrow
                arrow_len = 0.25
                hx = [robot.pose[0], robot.pose[0] + arrow_len * np.cos(robot.pose[2])]
                hy = [robot.pose[1], robot.pose[1] + arrow_len * np.sin(robot.pose[2])]
                plot_handles[rid]['heading'].set_data(hx, hy)
                
                # Bounding box
                hl, hw = CFG.robotLength / 2, CFG.robotWidth / 2
                corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
                cos_t, sin_t = np.cos(robot.pose[2]), np.sin(robot.pose[2])
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                rotated = (R @ corners.T).T + robot.pose[:2]
                plot_handles[rid]['box'].set_data(rotated[:, 0], rotated[:, 1])
                
                # Trajectory
                traj = np.array(robot.trajectory)
                plot_handles[rid]['traj'].set_data(traj[:, 0], traj[:, 1])
            
            title.set_text(f'Multi-Robot Simulation | t={sim_time:.1f}s | Reached: {num_reached}/{SIM.numRobots}')
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        # Check termination
        if num_reached == SIM.numRobots:
            print('\n✓ All robots reached their goals!')
            break
        
        # Advance time
        sim_time += SIM.dt
        frame_count += 1
        
        # Real-time pacing
        if SIM.realTimeMode:
            elapsed = time.time() - frame_start
            if elapsed < SIM.dt:
                time.sleep(SIM.dt - elapsed)
    
    # Results
    print('\n=== Simulation Complete ===')
    print(f'Total time: {sim_time:.1f} seconds')
    for rid, robot in robots.items():
        if robot.reached:
            print(f'  {rid}: Reached in {robot.reachedTime:.1f}s ({robot.replanCount} replans)')
        else:
            print(f'  {rid}: Did not reach goal')
    
    if SIM.realTimeMode:
        plt.ioff()
    plt.show()

if __name__ == '__main__':
    multi_robot_simulator()
