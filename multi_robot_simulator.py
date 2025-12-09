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
    reverseVel: float = 0.15  # Reverse speed when backing up
    minVelForTurning: float = 0.05  # Minimum velocity required to turn (car-like constraint)
    
    # Robot Dimensions (car-like)
    robotLength: float = 0.40  # Wheelbase
    robotWidth: float = 0.30
    safetyBuffer: float = 0.15
    collisionBuffer: float = 0.12  # Additional buffer for collision prevention
    hardCollisionRadius: float = 0.25  # Hard boundary circle from center (NEVER intersect)
    
    # Dubins Parameters
    maxSteeringAngle: float = np.deg2rad(35)
    
    # Path planning - SWITCHED TO A* for speed
    replanInterval: float = 0.6  # Replanning interval
    gridResolution: float = 0.15  # Grid cell size for A*
    
    # Dynamic obstacle parameters
    predictionHorizon: float = 2.5
    
    # Local Collision Avoidance - ENHANCED
    coordinationHorizon: float = 2.0
    safetyDistance: float = 0.55  # Minimum safe distance between robot centers
    emergencyDistance: float = 0.52  # Emergency stop distance
    decisionDistance: float = 0.8  # Distance at which to make reverse decisions
    
    # Cost-based reversing parameters
    reverseCostPerMeter: float = 2.0  # Cost multiplier for reversing (vs forward motion)
    waitCostPerSecond: float = 0.5  # Cost of waiting in place per second
    progressBenefit: float = 1.0  # Benefit of making progress toward goal
    reverseMinBenefit: float = 1.5  # Minimum cost savings to justify reversing
    commitmentTime: float = 2.0  # Once reversing, commit for this time (prevents oscillation)
    maxReverseDistance: float = 0.5  # Maximum distance to reverse
    
    # Map boundaries
    mapBoundaryMargin: float = 0.3  # Don't reverse if within this margin of boundary
    
    # Traffic jam detection and resolution
    stuckThreshold: int = 90  # Frames before considering stuck (3 seconds at 30Hz)
    trafficJamTimeout: float = 10.0  # Max time to wait before declaring traffic jam
    deadlockResetTime: float = 15.0  # Reset all robots' reverse decisions if stuck this long
    
    # Tolerances
    posTolerance: float = 0.25
    headingTolerance: float = np.deg2rad(30)
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.minTurnRadius = self.robotLength / np.tan(self.maxSteeringAngle)
        # Clearance radius: diagonal of box + safety buffer + collision buffer
        R_diag = np.sqrt(self.robotWidth**2 + self.robotLength**2) / 2.0
        self.robotClearanceRadius = R_diag + self.safetyBuffer + self.collisionBuffer
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
        
        # Cost-based reversing state
        self.reverseDecision = None  # 'forward', 'reverse', or 'wait'
        self.decisionTime = -np.inf  # When was the decision made
        self.reverseTargetDistance = 0.0  # How far to reverse
        self.reverseStartPos = initial_pose[:2].copy()
        
        # Traffic jam detection
        self.waitingTime = 0.0
        self.lastWaitPosition = initial_pose[:2].copy()
        self.inTrafficJam = False
        self.totalWaitTime = 0.0  # Total time spent waiting

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
    """Grid-based A* planner with road width constraint for robot clearance."""
    
    def __init__(self, map_size: np.ndarray, resolution: float, cfg: Config):
        self.map_size = map_size
        self.resolution = resolution
        self.cfg = cfg
        
        self.grid_width = int(np.ceil(map_size[0] / resolution))
        self.grid_height = int(np.ceil(map_size[1] / resolution))
        
        # Calculate road width in grid cells (robot width + safety margins)
        # Need clearance on both sides of the centerline
        road_width_meters = cfg.robotWidth + 2 * cfg.safetyBuffer
        self.road_half_width_cells = int(np.ceil(road_width_meters / (2 * resolution)))
        
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
            # Current position - use hard collision radius for obstacles
            cx, cy = self.world_to_grid(robot['x'], robot['y'])
            if self.is_valid(cx, cy):
                # Use the hard collision radius as the minimum
                hard_inflation = int(np.ceil(self.cfg.hardCollisionRadius / self.resolution))
                for dx in range(-hard_inflation, hard_inflation + 1):
                    for dy in range(-hard_inflation, hard_inflation + 1):
                        gx, gy = cx + dx, cy + dy
                        if self.is_valid(gx, gy):
                            wx, wy = self.grid_to_world(gx, gy)
                            dist = np.sqrt((wx - robot['x'])**2 + (wy - robot['y'])**2)
                            if dist < self.cfg.hardCollisionRadius:
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
    """Pure Pursuit controller with Dubins constraints - car-like behavior."""
    lookahead_x, lookahead_y, _ = find_lookahead_point(
        robot.pose[0], robot.pose[1], robot.path, cfg.lookaheadDist
    )
    
    # Calculate desired heading
    dx = lookahead_x - robot.pose[0]
    dy = lookahead_y - robot.pose[1]
    desired_heading = np.arctan2(dy, dx)
    heading_error = wrap_to_pi(desired_heading - robot.pose[2])
    
    # Check if target is ahead or behind
    is_forward = np.abs(heading_error) < np.deg2rad(90)
    
    # Linear velocity (always positive for forward, will be negated for reverse in main loop)
    if is_forward:
        # Target is ahead - normal forward motion
        lin_vel = cfg.targetLinearVel * max(0.3, np.cos(heading_error))
    else:
        # Target is behind - need to turn around
        # For car-like robots, we must creep forward while turning sharply
        lin_vel = cfg.targetLinearVel * 0.2  # Slow creep to enable turning
    
    # Angular velocity (only allowed when moving)
    dist_to_lookahead = np.sqrt(dx**2 + dy**2)
    if dist_to_lookahead > 1e-3 and lin_vel > cfg.minVelForTurning:
        # Standard pure pursuit curvature
        ang_vel = (2 * lin_vel * np.sin(heading_error)) / dist_to_lookahead
    else:
        # Not moving fast enough to turn (car-like constraint)
        ang_vel = 0.0
    
    # Apply Dubins constraints
    if lin_vel > cfg.minVelForTurning:
        max_ang_vel = lin_vel / cfg.minTurnRadius
    else:
        # If barely moving, can't turn at all
        max_ang_vel = 0.0
        ang_vel = 0.0
    
    ang_vel = np.clip(ang_vel, -max_ang_vel, max_ang_vel)
    lin_vel = np.clip(lin_vel, 0.0, cfg.maxLinVel)
    
    return lin_vel, ang_vel

# === LOCAL COLLISION AVOIDANCE ===

def check_circle_collision(pos1: np.ndarray, pos2: np.ndarray, radius1: float, radius2: float) -> bool:
    """Check if two circles collide."""
    dist = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    return dist < (radius1 + radius2)

def check_any_collision(pose1: np.ndarray, pose2: np.ndarray, cfg: Config) -> bool:
    """
    Check if two robots collide using BOTH circle and box collision.
    Returns True if EITHER the hard collision circles OR the bounding boxes intersect.
    """
    # Check hard collision circles first (fastest check)
    if check_circle_collision(pose1[:2], pose2[:2], cfg.hardCollisionRadius, cfg.hardCollisionRadius):
        return True
    
    # Check bounding box collision
    return check_box_collision(pose1, pose2, cfg)

def is_near_boundary(pose: np.ndarray, map_size: np.ndarray, margin: float) -> bool:
    """Check if robot is near map boundary."""
    x, y = pose[0], pose[1]
    return (x < margin or x > map_size[0] - margin or 
            y < margin or y > map_size[1] - margin)

def is_reverse_path_clear(my_pose: np.ndarray, all_robots: Dict[str, RobotState], my_id: str, cfg: Config, map_size: np.ndarray, reverse_dist: float = 0.4) -> bool:
    """
    Check if the path behind the robot is clear for reversing.
    Samples points along the reverse trajectory and checks map boundaries.
    """
    # Calculate reverse direction (opposite of current heading)
    reverse_heading = my_pose[2] + np.pi
    
    # Check multiple points along reverse path
    num_checks = 8
    for i in range(1, num_checks + 1):
        t = (i / num_checks) * reverse_dist
        check_x = my_pose[0] + t * np.cos(reverse_heading)
        check_y = my_pose[1] + t * np.sin(reverse_heading)
        check_pose = np.array([check_x, check_y, my_pose[2]])
        
        # Check if we'd go out of bounds
        if check_x < cfg.mapBoundaryMargin or check_x > map_size[0] - cfg.mapBoundaryMargin:
            return False
        if check_y < cfg.mapBoundaryMargin or check_y > map_size[1] - cfg.mapBoundaryMargin:
            return False
        
        # Check against all other robots (including parked)
        for rid, other_robot in all_robots.items():
            if rid == my_id:
                continue
            
            # Check collision with this point
            if check_any_collision(check_pose, other_robot.pose, cfg):
                return False
            
            # Also check center distance
            dist = np.sqrt((check_x - other_robot.pose[0])**2 + (check_y - other_robot.pose[1])**2)
            if dist < cfg.safetyDistance:
                return False
    
    return True

def check_trajectory_collision(my_pose: np.ndarray, my_vel: float, other_pose: np.ndarray, other_vel: np.ndarray, cfg: Config, time_horizon: float = 2.0, num_checks: int = 10) -> bool:
    """
    Check if the trajectory of two robots will intersect within time_horizon.
    Returns True if collision is predicted.
    """
    for i in range(1, num_checks + 1):
        t = (i / num_checks) * time_horizon
        
        # Predict my position
        my_future_x = my_pose[0] + my_vel * np.cos(my_pose[2]) * t
        my_future_y = my_pose[1] + my_vel * np.sin(my_pose[2]) * t
        my_future_pose = np.array([my_future_x, my_future_y, my_pose[2]])
        
        # Predict other position
        other_future_x = other_pose[0] + other_vel[0] * t
        other_future_y = other_pose[1] + other_vel[1] * t
        other_future_pose = np.array([other_future_x, other_future_y, other_pose[2]])
        
        # Check collision at this time step - use comprehensive check
        if check_any_collision(my_future_pose, other_future_pose, cfg):
            return True
        
        # Also check center-to-center distance
        dist = np.sqrt((my_future_x - other_future_x)**2 + (my_future_y - other_future_y)**2)
        if dist < cfg.safetyDistance:
            return True
    
    return False

def detect_traffic_jam(robots: Dict[str, RobotState], cfg: Config, sim_time: float) -> List[str]:
    """
    Detect which robots are in a traffic jam and check for global deadlock.
    Returns list of robot IDs that should be given priority to clear the jam.
    """
    jammed_robots = []
    all_waiting = True
    
    for rid, robot in robots.items():
        if robot.reached:
            continue
        
        # Check if making progress
        if robot.waitingTime < cfg.stuckThreshold * 0.033:  # Convert frames to seconds
            all_waiting = False
        
        # Check if robot has been waiting/stuck for too long
        if robot.waitingTime > cfg.trafficJamTimeout:
            jammed_robots.append((rid, robot))
    
    if not jammed_robots:
        return []
    
    # Check for global deadlock - all robots waiting for a long time
    if all_waiting and sim_time > cfg.deadlockResetTime:
        # Global deadlock detected - reset all robots' decisions
        for robot in robots.values():
            if not robot.reached:
                robot.reverseDecision = None
                robot.decisionTime = -np.inf
                robot.waitingTime = 0.0
        print(f'  [{sim_time:.1f}s] Global deadlock detected - resetting all decisions')
        return []
    
    # Sort by total wait time and distance to goal
    jammed_robots.sort(key=lambda x: (
        -x[1].totalWaitTime,  # More wait time = higher priority (negative for descending)
        np.sqrt((x[1].pose[0] - x[1].goal[0])**2 + (x[1].pose[1] - x[1].goal[1])**2)
    ))
    
    # Return top 2 robots that deserve priority
    return [jammed_robots[i][0] for i in range(min(2, len(jammed_robots)))]

def calculate_reverse_cost_benefit(my_robot: RobotState, other_robot: RobotState, my_id: str, cfg: Config, all_robots: Dict[str, RobotState], map_size: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the cost-benefit of reversing vs waiting.
    Returns: (reverse_cost, wait_cost) - lower is better
    """
    my_pose = my_robot.pose
    my_dist_to_goal = np.sqrt((my_pose[0] - my_robot.goal[0])**2 + (my_pose[1] - my_robot.goal[1])**2)
    other_dist_to_goal = np.sqrt((other_robot.pose[0] - other_robot.goal[0])**2 + 
                                  (other_robot.pose[1] - other_robot.goal[1])**2)
    
    # Estimate how far I'd need to reverse to clear the path
    curr_dist = np.sqrt((my_pose[0] - other_robot.pose[0])**2 + (my_pose[1] - other_robot.pose[1])**2)
    estimated_reverse_dist = max(0.3, cfg.safetyDistance - curr_dist + 0.2)  # Add margin
    estimated_reverse_dist = min(estimated_reverse_dist, cfg.maxReverseDistance)
    
    # Check if reverse path is clear for this distance
    if not is_reverse_path_clear(my_pose, all_robots, my_id, cfg, map_size, estimated_reverse_dist):
        return np.inf, 0.0  # Can't reverse - infinite cost
    
    # Check boundary
    if is_near_boundary(my_pose, map_size, cfg.mapBoundaryMargin):
        return np.inf, 0.0  # Can't reverse near boundary
    
    # Calculate reverse cost
    # Cost = distance * cost_per_meter + opportunity cost (lose progress toward goal)
    reverse_cost = (estimated_reverse_dist * cfg.reverseCostPerMeter + 
                    estimated_reverse_dist / cfg.targetLinearVel * cfg.waitCostPerSecond)
    
    # Calculate wait cost
    # Estimate how long other robot will take to clear based on their speed and distance
    other_speed = np.linalg.norm(other_robot.velocity)
    if other_speed > 0.05:
        # Other is moving - estimate time to clear
        clearance_dist = curr_dist + cfg.safetyDistance
        estimated_wait_time = clearance_dist / other_speed
    else:
        # Other is stopped - assume they'll wait indefinitely
        estimated_wait_time = 10.0  # Large penalty for waiting on stopped robot
    
    wait_cost = estimated_wait_time * cfg.waitCostPerSecond
    
    # Adjust costs based on priority (distance to goal)
    # If I'm much closer to goal, increase the wait cost (I should go)
    # If other is much closer, decrease the wait cost (I should wait)
    priority_ratio = other_dist_to_goal / (my_dist_to_goal + 0.01)
    
    if priority_ratio > 1.5:  # Other is 50%+ farther
        # I'm much closer - increase wait cost
        wait_cost *= 1.5
    elif priority_ratio < 0.67:  # I'm 50%+ farther
        # Other is much closer - decrease wait cost (I should wait)
        wait_cost *= 0.5
    
    return reverse_cost, wait_cost

def negotiate_velocity(my_robot: RobotState, all_robots: Dict[str, RobotState], my_id: str, cfg: Config, v_ideal: float, priority_robots: List[str], sim_time: float, map_size: np.ndarray) -> Tuple[float, str, float]:
    """
    Cost-based collision avoidance with commitment to decisions.
    Returns: (commanded_velocity, decision, target_reverse_distance)
    decision is 'forward', 'reverse', or 'wait'
    """
    my_pose = my_robot.pose
    my_dist_to_goal = np.sqrt((my_pose[0] - my_robot.goal[0])**2 + (my_pose[1] - my_robot.goal[1])**2)
    
    i_have_priority = my_id in priority_robots
    
    # Check if we're in a committed decision
    time_since_decision = sim_time - my_robot.decisionTime
    if my_robot.reverseDecision is not None and time_since_decision < cfg.commitmentTime:
        # Still committed to previous decision
        if my_robot.reverseDecision == 'reverse':
            # Check if we've reversed enough
            reverse_dist = np.linalg.norm(my_robot.pose[:2] - my_robot.reverseStartPos)
            if reverse_dist >= my_robot.reverseTargetDistance:
                # Done reversing
                my_robot.reverseDecision = None
                return 0.0, 'wait', 0.0
            else:
                # Continue reversing
                return -cfg.reverseVel, 'reverse', my_robot.reverseTargetDistance
        elif my_robot.reverseDecision == 'forward':
            # Continue forward cautiously
            return v_ideal * 0.6, 'forward', 0.0
        else:  # 'wait'
            return 0.0, 'wait', 0.0
    
    # Decision commitment expired or no decision - make new decision
    my_robot.reverseDecision = None
    
    # Collect conflicts
    conflicts = []
    
    for rid, other_robot in all_robots.items():
        if rid == my_id:
            continue
        
        other_pose = other_robot.pose
        curr_dist = np.sqrt((my_pose[0] - other_pose[0])**2 + (my_pose[1] - other_pose[1])**2)
        
        # ABSOLUTE HARD STOP - collision imminent
        if check_any_collision(my_pose, other_pose, cfg):
            return 0.0, 'wait', 0.0
        
        # Hard distance check
        if curr_dist < cfg.safetyDistance:
            return 0.0, 'wait', 0.0
        
        # Only consider conflicts within decision distance
        if curr_dist < cfg.decisionDistance:
            other_dist_to_goal = np.sqrt((other_pose[0] - other_robot.goal[0])**2 + 
                                         (other_pose[1] - other_robot.goal[1])**2) if not other_robot.reached else np.inf
            
            conflicts.append({
                'rid': rid,
                'robot': other_robot,
                'distance': curr_dist,
                'dist_to_goal': other_dist_to_goal,
                'is_parked': other_robot.reached
            })
    
    # No conflicts - go forward
    if not conflicts:
        return v_ideal, 'forward', 0.0
    
    # Sort by distance
    conflicts.sort(key=lambda x: x['distance'])
    
    # Find critical conflict
    critical_conflict = None
    for conflict in conflicts:
        # Parked robots - just avoid
        if conflict['is_parked']:
            if conflict['distance'] < cfg.emergencyDistance + 0.1:
                return 0.0, 'wait', 0.0
            continue
        
        # Check trajectory collision
        will_collide = check_trajectory_collision(
            my_pose, v_ideal, conflict['robot'].pose, conflict['robot'].velocity,
            cfg, cfg.coordinationHorizon
        )
        
        if will_collide or conflict['distance'] < cfg.emergencyDistance:
            critical_conflict = conflict
            break
    
    # No critical conflicts
    if critical_conflict is None:
        return v_ideal * 0.7, 'forward', 0.0
    
    # We have a critical conflict - calculate costs
    other_robot = critical_conflict['robot']
    
    # If other robot is stopped, try to go around
    other_speed = np.linalg.norm(other_robot.velocity)
    if other_speed < 0.05:
        return v_ideal * 0.3, 'forward', 0.0
    
    # Traffic jam priority
    if i_have_priority:
        my_robot.reverseDecision = 'forward'
        my_robot.decisionTime = sim_time
        return v_ideal * 0.5, 'forward', 0.0
    
    # Calculate cost-benefit
    reverse_cost, wait_cost = calculate_reverse_cost_benefit(
        my_robot, other_robot, my_id, cfg, all_robots, map_size
    )
    
    # Make decision based on costs
    if reverse_cost < wait_cost - cfg.reverseMinBenefit:
        # Reversing is significantly better - commit to it
        estimated_reverse_dist = max(0.3, cfg.safetyDistance - critical_conflict['distance'] + 0.2)
        estimated_reverse_dist = min(estimated_reverse_dist, cfg.maxReverseDistance)
        
        my_robot.reverseDecision = 'reverse'
        my_robot.decisionTime = sim_time
        my_robot.reverseTargetDistance = estimated_reverse_dist
        my_robot.reverseStartPos = my_robot.pose[:2].copy()
        
        return -cfg.reverseVel, 'reverse', estimated_reverse_dist
    else:
        # Waiting is better or costs are similar
        my_robot.reverseDecision = 'wait'
        my_robot.decisionTime = sim_time
        return 0.0, 'wait', 0.0

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
    robotIds = list(robots.keys())  # Store robot IDs for color indexing
    
    for i, (rid, robot) in enumerate(robots.items()):
        col = colors(i)
        plot_handles[rid] = {
            'goal': ax.plot(robot.goal[0], robot.goal[1], 'x', color=col, markersize=12, markeredgewidth=3)[0],
            'path': ax.plot([], [], '--', color=col, linewidth=1, alpha=0.4)[0],
            'traj': ax.plot([], [], '-', color=col, linewidth=1.5, alpha=0.7)[0],
            'robot': ax.plot([], [], 'o', color=col, markersize=10)[0],
            'heading': ax.plot([], [], '-', color=col, linewidth=2)[0],
            'box': ax.plot([], [], '-', color=col, linewidth=1.5)[0],
            'hardCircle': ax.plot([], [], ':', color=col, linewidth=2, alpha=0.6)[0]  # Hard collision circle
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
        
        # Detect traffic jams and assign priority
        priority_robots = detect_traffic_jam(robots, CFG, sim_time)
        
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
                robot.waitingTime += SIM.dt
                robot.totalWaitTime += SIM.dt
            else:
                robot.stuckCounter = 0
                robot.waitingTime = 0.0
                robot.inTrafficJam = False
                
            robot.lastPosition = robot.pose[:2].copy()
            
            # Mark if in traffic jam
            if robot.waitingTime > CFG.trafficJamTimeout:
                robot.inTrafficJam = True
            
            # Replan if needed
            should_replan = (
                (sim_time - robot.lastReplanTime) >= CFG.replanInterval or
                robot.stuckCounter > CFG.stuckThreshold or
                robot.inTrafficJam
            )
            
            if should_replan:
                # Create obstacle map from other robots
                other_robots_list = []
                for other_id, other_robot in robots.items():
                    if other_id != rid:  # Include ALL other robots (moving AND parked)
                        other_robots_list.append({
                            'x': other_robot.pose[0],
                            'y': other_robot.pose[1],
                            'vx': other_robot.velocity[0] if not other_robot.reached else 0.0,
                            'vy': other_robot.velocity[1] if not other_robot.reached else 0.0,
                            'reached': other_robot.reached  # Track if parked
                        })
                
                obstacle_map = planner.create_obstacle_map(other_robots_list)
                
                # For car-like robots, the start heading matters
                # A* doesn't account for heading, so we accept this limitation
                # In production, would use Hybrid A* or RRT with Dubins curves
                newPath = planner.plan(robot.pose, robot.goal, obstacle_map)
                
                if newPath is not None:
                    robot.path = newPath
                    robot.replanCount += 1
                    robot.stuckCounter = 0
                    plot_handles[rid]['path'].set_data(robot.path.x, robot.path.y)
                else:
                    # No path found - calculate if reversing would help
                    if is_reverse_path_clear(robot.pose, robots, rid, CFG, SIM.mapSize, 0.3):
                        # Make a reverse decision with short distance
                        robot.reverseDecision = 'reverse'
                        robot.decisionTime = sim_time
                        robot.reverseTargetDistance = 0.3
                        robot.reverseStartPos = robot.pose[:2].copy()
                
                robot.lastReplanTime = sim_time
            
            # Pure pursuit control
            lin_vel, ang_vel = pure_pursuit_control(robot, CFG)
            
            # Collision avoidance and decision making
            v_cmd, decision, target_reverse_dist = negotiate_velocity(
                robot, robots, rid, CFG, lin_vel, priority_robots, sim_time, SIM.mapSize
            )
            
            # Execute decision
            if decision == 'reverse':
                lin_vel = -CFG.reverseVel
                ang_vel = 0.0  # No turning while reversing
                
                # Safety check while reversing
                if not is_reverse_path_clear(robot.pose, robots, rid, CFG, SIM.mapSize, CFG.reverseVel * SIM.dt * 2):
                    robot.reverseDecision = None  # Abort reverse
                    lin_vel = 0.0
            elif decision == 'wait':
                lin_vel = 0.0
                ang_vel = 0.0
            else:  # 'forward'
                lin_vel = v_cmd
                # ang_vel already set by pure pursuit
            
            # Car-like constraint: Can only turn when moving
            if abs(lin_vel) < CFG.minVelForTurning:
                ang_vel = 0.0
            
            # Update state
            robot.pose[0] += lin_vel * np.cos(robot.pose[2]) * SIM.dt
            robot.pose[1] += lin_vel * np.sin(robot.pose[2]) * SIM.dt
            robot.pose[2] = wrap_to_pi(robot.pose[2] + ang_vel * SIM.dt)
            
            robot.trajectory.append(robot.pose[:2].copy())
        
        # Update visualization
        if frame_count % 2 == 0 or not SIM.realTimeMode:
            for rid, robot in robots.items():
                col = colors(robotIds.index(rid))
                
                if robot.reached:
                    # Keep showing parked robots in a dimmed state
                    plot_handles[rid]['robot'].set_data([robot.pose[0]], [robot.pose[1]])
                    plot_handles[rid]['robot'].set_markerfacecolor(col)
                    plot_handles[rid]['robot'].set_markeredgecolor(col)
                    plot_handles[rid]['robot'].set_alpha(0.3)  # Dim the robot
                    
                    # Show the box dimmed
                    hl, hw = CFG.robotLength / 2, CFG.robotWidth / 2
                    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
                    cos_t, sin_t = np.cos(robot.pose[2]), np.sin(robot.pose[2])
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    rotated = (R @ corners.T).T + robot.pose[:2]
                    plot_handles[rid]['box'].set_data(rotated[:, 0], rotated[:, 1])
                    plot_handles[rid]['box'].set_alpha(0.3)  # Dim the box
                    
                    # Show hard collision circle dimmed
                    theta_circle = np.linspace(0, 2 * np.pi, 30)
                    circleX = robot.pose[0] + CFG.hardCollisionRadius * np.cos(theta_circle)
                    circleY = robot.pose[1] + CFG.hardCollisionRadius * np.sin(theta_circle)
                    plot_handles[rid]['hardCircle'].set_data(circleX, circleY)
                    plot_handles[rid]['hardCircle'].set_alpha(0.2)  # Dim the circle
                    
                    plot_handles[rid]['heading'].set_data([], [])  # Hide heading arrow
                    continue
                
                # Active robot visualization
                # Robot position - change color if reversing
                plot_handles[rid]['robot'].set_data([robot.pose[0]], [robot.pose[1]])
                plot_handles[rid]['robot'].set_alpha(1.0)  # Full opacity
                if robot.reverseDecision == 'reverse':
                    # Make robot red when reversing
                    plot_handles[rid]['robot'].set_markerfacecolor('red')
                    plot_handles[rid]['robot'].set_markeredgecolor('red')
                else:
                    # Normal color
                    plot_handles[rid]['robot'].set_markerfacecolor(col)
                    plot_handles[rid]['robot'].set_markeredgecolor(col)
                
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
                plot_handles[rid]['box'].set_alpha(1.0)  # Full opacity
                
                # Hard collision circle (0.25m radius) - dotted line
                theta_circle = np.linspace(0, 2 * np.pi, 30)
                circleX = robot.pose[0] + CFG.hardCollisionRadius * np.cos(theta_circle)
                circleY = robot.pose[1] + CFG.hardCollisionRadius * np.sin(theta_circle)
                plot_handles[rid]['hardCircle'].set_data(circleX, circleY)
                plot_handles[rid]['hardCircle'].set_alpha(0.6)  # Semi-transparent
                
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
