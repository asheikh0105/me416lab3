import numpy as np
import time
import paho.mqtt.client as mqtt
from enum import Enum
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import socket
import struct
from threading import Thread, Lock
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.animation import FuncAnimation


# === ROBOT STATE MACHINE ===

class RobotState(Enum):
    """Robot run states as defined in the requirements"""
    WAIT = "WAIT"
    GO = "GO"
    STOP = "STOP"
    HALT = "HALT"

# === CONFIGURATION ===

@dataclass
class Config:
    """Controller and Planning Configuration Parameters."""
    # Pure Pursuit Controller
    lookaheadDist: float = 0.4
    targetLinearVel: float = 0.3
    maxLinVel: float = 0.3
    maxAngVel: float = np.deg2rad(90)
    reverseVel: float = 0.15
    minVelForTurning: float = 0.05
    
    # Robot Dimensions (car-like)
    robotLength: float = 0.40
    robotWidth: float = 0.30
    safetyBuffer: float = 0.15
    collisionBuffer: float = 0.12
    hardCollisionRadius: float = 0.25
    
    # Dubins Parameters
    maxSteeringAngle: float = np.deg2rad(35)
    
    # Path planning
    replanInterval: float = 0.6
    gridResolution: float = 0.15
    
    # Dynamic obstacle parameters
    predictionHorizon: float = 2.5
    velocityHistorySize: int = 5
    
    # Local Collision Avoidance
    coordinationHorizon: float = 2.0
    safetyDistance: float = 0.55
    emergencyDistance: float = 0.52
    decisionDistance: float = 0.8
    
    # Cost-based reversing parameters
    reverseCostPerMeter: float = 2.0
    waitCostPerSecond: float = 0.5
    progressBenefit: float = 1.0
    reverseMinBenefit: float = 1.5
    commitmentTime: float = 2.0
    maxReverseDistance: float = 0.5
    
    # Map boundaries - ARENA MoCap room
    # Origin at center, x: [-4.5, 4.5]m (width, toward hallway), y: [-2.5, 2.5]m (depth, toward street)
    mapSize: np.ndarray = np.array([9.0, 5.0])  # 9m x 5m
    mapOrigin: np.ndarray = np.array([-4.5, -2.5])  # Bottom-left corner in world frame
    mapBoundaryMargin: float = 0.3
    
    # Stuck detection
    stuckThreshold: int = 90
    stuckDistanceThreshold: float = 0.01
    
    # Tolerances
    posTolerance: float = 0.25
    headingTolerance: float = np.deg2rad(30)
    
    # Control loop rate
    controlRate: float = 30.0  # Hz
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.minTurnRadius = self.robotLength / np.tan(self.maxSteeringAngle)
        R_diag = np.sqrt(self.robotWidth**2 + self.robotLength**2) / 2.0
        self.robotClearanceRadius = R_diag + self.safetyBuffer + self.collisionBuffer
        self.safeSeparation = 2 * self.robotClearanceRadius
        self.dt = 1.0 / self.controlRate

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

class ObstacleState:
    """Represents a dynamic obstacle (other robot) from mocap."""
    def __init__(self, obstacle_id: str, x: float, y: float, theta: float = 0.0):
        self.id = obstacle_id
        self.pose = np.array([x, y, theta])
        self.velocity = np.array([0.0, 0.0])
        self.position_history = deque(maxlen=5)
        self.position_history.append(np.array([x, y]))
        self.last_update_time = time.time()
    
    def update(self, x: float, y: float, theta: float = 0.0):
        """Update obstacle state with new mocap data."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        self.position_history.append(np.array([x, y]))
        
        if len(self.position_history) >= 2 and dt > 0:
            total_dx = self.position_history[-1][0] - self.position_history[0][0]
            total_dy = self.position_history[-1][1] - self.position_history[0][1]
            total_dt = dt * (len(self.position_history) - 1)
            
            self.velocity[0] = total_dx / total_dt if total_dt > 0 else 0.0
            self.velocity[1] = total_dy / total_dt if total_dt > 0 else 0.0
        
        self.pose = np.array([x, y, theta])
        self.last_update_time = current_time

# === HELPER FUNCTIONS ===

def wrap_to_pi(angle: float) -> float:
    """Wraps an angle to the range (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def check_box_collision(pose1: np.ndarray, pose2: np.ndarray, cfg: Config) -> bool:
    """Check if two oriented bounding boxes collide."""
    x1, y1, theta1 = pose1
    x2, y2, theta2 = pose2
    
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if dist > (cfg.robotLength + cfg.robotWidth):
        return False
    
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
    
    axes = [
        np.array([np.cos(theta1), np.sin(theta1)]),
        np.array([-np.sin(theta1), np.cos(theta1)]),
        np.array([np.cos(theta2), np.sin(theta2)]),
        np.array([-np.sin(theta2), np.cos(theta2)])
    ]
    
    for axis in axes:
        proj1 = corners1 @ axis
        proj2 = corners2 @ axis
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False
    
    return True

def check_circle_collision(pos1: np.ndarray, pos2: np.ndarray, radius1: float, radius2: float) -> bool:
    """Check if two circles collide."""
    dist = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    return dist < (radius1 + radius2)

def check_any_collision(pose1: np.ndarray, pose2: np.ndarray, cfg: Config) -> bool:
    """Check collision using both circle and box."""
    if check_circle_collision(pose1[:2], pose2[:2], cfg.hardCollisionRadius, cfg.hardCollisionRadius):
        return True
    return check_box_collision(pose1, pose2, cfg)

def is_near_boundary(pose: np.ndarray, map_size: np.ndarray, margin: float, map_origin: np.ndarray = np.array([-4.5, -2.5])) -> bool:
    """Check if robot is near map boundary."""
    x, y = pose[0], pose[1]
    # Convert from center-origin coords to boundary checks
    x_max = map_origin[0] + map_size[0]
    y_max = map_origin[1] + map_size[1]
    
    return (x < map_origin[0] + margin or x > x_max - margin or 
            y < map_origin[1] + margin or y > y_max - margin)

def check_trajectory_collision(my_pose: np.ndarray, my_vel: float, other_pose: np.ndarray, 
                               other_vel: np.ndarray, cfg: Config, time_horizon: float = 2.0, 
                               num_checks: int = 10) -> bool:
    """Check if trajectories will intersect."""
    for i in range(1, num_checks + 1):
        t = (i / num_checks) * time_horizon
        
        my_future_x = my_pose[0] + my_vel * np.cos(my_pose[2]) * t
        my_future_y = my_pose[1] + my_vel * np.sin(my_pose[2]) * t
        my_future_pose = np.array([my_future_x, my_future_y, my_pose[2]])
        
        other_future_x = other_pose[0] + other_vel[0] * t
        other_future_y = other_pose[1] + other_vel[1] * t
        other_future_pose = np.array([other_future_x, other_future_y, other_pose[2]])
        
        if check_any_collision(my_future_pose, other_future_pose, cfg):
            return True
        
        dist = np.sqrt((my_future_x - other_future_x)**2 + (my_future_y - other_future_y)**2)
        if dist < cfg.safetyDistance:
            return True
    
    return False

def find_lookahead_point(robot_x: float, robot_y: float, path: Path, 
                         lookahead_dist: float) -> Tuple[float, float, float]:
    """Find lookahead point on path."""
    distances = np.sqrt((path.x - robot_x)**2 + (path.y - robot_y)**2)
    closest_idx = np.argmin(distances)
    cross_track_error = distances[closest_idx]
    
    s_closest = path.s[closest_idx]
    s_target = s_closest + lookahead_dist
    
    if s_target >= path.s[-1]:
        return path.x[-1], path.y[-1], cross_track_error
    
    idx = np.searchsorted(path.s, s_target)
    if idx >= len(path.x):
        idx = len(path.x) - 1
    
    return path.x[idx], path.y[idx], cross_track_error

# === A* PLANNER ===

class AStarPlanner:
    """Grid-based A* planner."""
    
    def __init__(self, map_size: np.ndarray, resolution: float, cfg: Config):
        self.map_size = map_size
        self.map_origin = cfg.mapOrigin
        self.resolution = resolution
        self.cfg = cfg
        
        # Grid dimensions based on absolute map size
        self.grid_width = int(np.ceil(map_size[0] / resolution))
        self.grid_height = int(np.ceil(map_size[1] / resolution))
        
        self.motions = [
            [1, 0, 1.0], [0, 1, 1.0], [-1, 0, 1.0], [0, -1, 1.0],
            [1, 1, 1.414], [1, -1, 1.414], [-1, 1, 1.414], [-1, -1, 1.414]
        ]
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (center origin) to grid indices."""
        # Shift from center-origin to grid-origin (bottom-left)
        x_shifted = x - self.map_origin[0]
        y_shifted = y - self.map_origin[1]
        
        gx = int(np.floor(x_shifted / self.resolution))
        gy = int(np.floor(y_shifted / self.resolution))
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (center origin)."""
        # Grid to shifted coordinates
        x_shifted = (gx + 0.5) * self.resolution
        y_shifted = (gy + 0.5) * self.resolution
        
        # Shift back to center-origin
        x = x_shifted + self.map_origin[0]
        y = y_shifted + self.map_origin[1]
        return x, y
    
    def is_valid(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.grid_width and 0 <= gy < self.grid_height
    
    def create_obstacle_map(self, obstacles: List[Dict]) -> np.ndarray:
        """Create binary obstacle map."""
        obstacle_map = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        
        inflation_cells = int(np.ceil(self.cfg.robotClearanceRadius / self.resolution))
        
        for obs in obstacles:
            cx, cy = self.world_to_grid(obs['x'], obs['y'])
            if self.is_valid(cx, cy):
                hard_inflation = int(np.ceil(self.cfg.hardCollisionRadius / self.resolution))
                for dx in range(-hard_inflation, hard_inflation + 1):
                    for dy in range(-hard_inflation, hard_inflation + 1):
                        gx, gy = cx + dx, cy + dy
                        if self.is_valid(gx, gy):
                            wx, wy = self.grid_to_world(gx, gy)
                            dist = np.sqrt((wx - obs['x'])**2 + (wy - obs['y'])**2)
                            if dist < self.cfg.hardCollisionRadius:
                                obstacle_map[gx, gy] = True
            
            vx, vy = obs['vx'], obs['vy']
            speed = np.sqrt(vx**2 + vy**2)
            
            if speed > 0.05:
                pred_dist = speed * self.cfg.predictionHorizon
                num_preds = max(3, int(pred_dist / self.resolution))
                
                for i in range(1, num_preds + 1):
                    t = (i / num_preds) * self.cfg.predictionHorizon
                    px = obs['x'] + vx * t
                    py = obs['y'] + vy * t
                    
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
        
        open_set = {(start_gx, start_gy)}
        closed_set = set()
        
        g_score = {(start_gx, start_gy): 0}
        f_score = {(start_gx, start_gy): self._heuristic(start_gx, start_gy, goal_gx, goal_gy)}
        came_from = {}
        
        while open_set:
            current = min(open_set, key=lambda n: f_score.get(n, np.inf))
            
            if current == (goal_gx, goal_gy):
                path_grid = [current]
                while current in came_from:
                    current = came_from[current]
                    path_grid.append(current)
                path_grid.reverse()
                
                path_x = []
                path_y = []
                for gx, gy in path_grid:
                    wx, wy = self.grid_to_world(gx, gy)
                    path_x.append(wx)
                    path_y.append(wy)
                
                return Path(np.array(path_x), np.array(path_y))
            
            open_set.remove(current)
            closed_set.add(current)
            
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
        
        return None
    
    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# === ROBOT NAVIGATOR ===

class RobotNavigator:
    """Main navigation controller for a single robot."""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.planner = AStarPlanner(config.mapSize, config.gridResolution, config)
        
        # Robot state - pose will be updated from MoCap
        self.pose = np.array([0.0, 0.0, 0.0])  # Will be set by MoCap
        self.goal = None  # Will be set by MQTT goal command
        self.path = None
        
        # Control state
        self.lastReplanTime = -np.inf
        self.replanCount = 0
        self.stuckCounter = 0
        self.lastPosition = np.array([0.0, 0.0])
        
        # Reverse decision state
        self.reverseDecision = None
        self.decisionTime = -np.inf
        self.reverseTargetDistance = 0.0
        self.reverseStartPos = np.array([0.0, 0.0])
        
        # Obstacles from mocap
        self.obstacles: Dict[str, ObstacleState] = {}
        
        # Trajectory history
        self.trajectory = []
        
        # Status
        self.goal_reached = False
        
        print("Robot Navigator initialized")
        print("  Waiting for MoCap pose data...")
        print("  Waiting for goal from MQTT...")
    
    def set_pose(self, x: float, y: float, theta: float):
        """Update robot pose from mocap/odometry."""
        self.pose = np.array([x, y, theta])
        self.trajectory.append(np.array([x, y]))
        
        pos_change = np.linalg.norm(self.pose[:2] - self.lastPosition)
        if pos_change < self.cfg.stuckDistanceThreshold:
            self.stuckCounter += 1
        else:
            self.stuckCounter = 0
        self.lastPosition = self.pose[:2].copy()
    
    def set_goal(self, goal_x: float, goal_y: float):
        """Set navigation goal."""
        self.goal = np.array([goal_x, goal_y])
        self.goal_reached = False
        self.replanCount = 0
        print(f"New goal set: ({goal_x:.2f}, {goal_y:.2f})")
        
        path_x = np.linspace(self.pose[0], goal_x, 20)
        path_y = np.linspace(self.pose[1], goal_y, 20)
        self.path = Path(path_x, path_y)
    
    def update_obstacle(self, obstacle_id: str, x: float, y: float, theta: float = 0.0):
        """Update obstacle state from mocap data."""
        if obstacle_id not in self.obstacles:
            self.obstacles[obstacle_id] = ObstacleState(obstacle_id, x, y, theta)
        else:
            self.obstacles[obstacle_id].update(x, y, theta)
    
    def remove_obstacle(self, obstacle_id: str):
        """Remove an obstacle that's no longer tracked."""
        if obstacle_id in self.obstacles:
            del self.obstacles[obstacle_id]
    
    def clear_obstacles(self):
        """Clear all obstacles."""
        self.obstacles.clear()
    
    def compute_control(self) -> Tuple[float, float, Dict]:
        """Compute velocity commands for the robot."""
        current_time = time.time()
        
        if self.goal is None:
            return 0.0, 0.0, {'error': 'No goal set'}
        
        dist_to_goal = np.linalg.norm(self.pose[:2] - self.goal)
        if dist_to_goal < self.cfg.posTolerance:
            self.goal_reached = True
            return 0.0, 0.0, {
                'goal_reached': True,
                'distance_to_goal': dist_to_goal,
                'replan_count': self.replanCount
            }
        
        should_replan = (
            (current_time - self.lastReplanTime) >= self.cfg.replanInterval or
            self.stuckCounter > self.cfg.stuckThreshold or
            self.path is None
        )
        
        if should_replan:
            new_path = self._replan()
            if new_path is not None:
                self.path = new_path
                self.replanCount += 1
                self.stuckCounter = 0
            self.lastReplanTime = current_time
        
        if self.path is None:
            return 0.0, 0.0, {'error': 'No valid path'}
        
        lin_vel, ang_vel = self._pure_pursuit_control()
        v_cmd, decision, _ = self._negotiate_velocity(lin_vel, current_time)
        
        if decision == 'reverse':
            lin_vel = -self.cfg.reverseVel
            ang_vel = 0.0
            
            if not self._is_reverse_path_clear(self.cfg.reverseVel * self.cfg.dt * 2):
                self.reverseDecision = None
                lin_vel = 0.0
        elif decision == 'wait':
            lin_vel = 0.0
            ang_vel = 0.0
        else:
            lin_vel = v_cmd
        
        if abs(lin_vel) < self.cfg.minVelForTurning:
            ang_vel = 0.0
        
        status = {
            'goal_reached': False,
            'stuck': self.stuckCounter > self.cfg.stuckThreshold,
            'reversing': decision == 'reverse',
            'waiting': decision == 'wait',
            'distance_to_goal': dist_to_goal,
            'num_obstacles': len(self.obstacles),
            'replan_count': self.replanCount,
            'linear_velocity': lin_vel,
            'angular_velocity': ang_vel
        }
        
        return lin_vel, ang_vel, status
    
    def _replan(self) -> Optional[Path]:
        """Replan path avoiding obstacles."""
        obstacle_list = []
        for obs in self.obstacles.values():
            obstacle_list.append({
                'x': obs.pose[0],
                'y': obs.pose[1],
                'vx': obs.velocity[0],
                'vy': obs.velocity[1],
                'reached': False
            })
        
        obstacle_map = self.planner.create_obstacle_map(obstacle_list)
        return self.planner.plan(self.pose, self.goal, obstacle_map)
    
    def _pure_pursuit_control(self) -> Tuple[float, float]:
        """Pure Pursuit controller."""
        lookahead_x, lookahead_y, _ = find_lookahead_point(
            self.pose[0], self.pose[1], self.path, self.cfg.lookaheadDist
        )
        
        dx = lookahead_x - self.pose[0]
        dy = lookahead_y - self.pose[1]
        desired_heading = np.arctan2(dy, dx)
        heading_error = wrap_to_pi(desired_heading - self.pose[2])
        
        is_forward = np.abs(heading_error) < np.deg2rad(90)
        
        if is_forward:
            lin_vel = self.cfg.targetLinearVel * max(0.3, np.cos(heading_error))
        else:
            lin_vel = self.cfg.targetLinearVel * 0.2
        
        dist_to_lookahead = np.sqrt(dx**2 + dy**2)
        if dist_to_lookahead > 1e-3 and lin_vel > self.cfg.minVelForTurning:
            ang_vel = (2 * lin_vel * np.sin(heading_error)) / dist_to_lookahead
        else:
            ang_vel = 0.0
        
        if lin_vel > self.cfg.minVelForTurning:
            max_ang_vel = lin_vel / self.cfg.minTurnRadius
        else:
            max_ang_vel = 0.0
            ang_vel = 0.0
        
        ang_vel = np.clip(ang_vel, -max_ang_vel, max_ang_vel)
        lin_vel = np.clip(lin_vel, 0.0, self.cfg.maxLinVel)
        
        return lin_vel, ang_vel
    
    def _negotiate_velocity(self, v_ideal: float, current_time: float) -> Tuple[float, str, float]:
        """Collision avoidance with cost-based decisions."""
        time_since_decision = current_time - self.decisionTime
        if self.reverseDecision is not None and time_since_decision < self.cfg.commitmentTime:
            if self.reverseDecision == 'reverse':
                reverse_dist = np.linalg.norm(self.pose[:2] - self.reverseStartPos)
                if reverse_dist >= self.reverseTargetDistance:
                    self.reverseDecision = None
                    return 0.0, 'wait', 0.0
                else:
                    return -self.cfg.reverseVel, 'reverse', self.reverseTargetDistance
            elif self.reverseDecision == 'forward':
                return v_ideal * 0.6, 'forward', 0.0
            else:
                return 0.0, 'wait', 0.0
        
        self.reverseDecision = None
        
        conflicts = []
        for obs in self.obstacles.values():
            curr_dist = np.linalg.norm(self.pose[:2] - obs.pose[:2])
            
            if check_any_collision(self.pose, obs.pose, self.cfg):
                return 0.0, 'wait', 0.0
            
            if curr_dist < self.cfg.safetyDistance:
                return 0.0, 'wait', 0.0
            
            if curr_dist < self.cfg.decisionDistance:
                conflicts.append({
                    'obstacle': obs,
                    'distance': curr_dist
                })
        
        if not conflicts:
            return v_ideal, 'forward', 0.0
        
        conflicts.sort(key=lambda x: x['distance'])
        critical_conflict = None
        
        for conflict in conflicts:
            will_collide = check_trajectory_collision(
                self.pose, v_ideal, conflict['obstacle'].pose, 
                conflict['obstacle'].velocity, self.cfg, self.cfg.coordinationHorizon
            )
            
            if will_collide or conflict['distance'] < self.cfg.emergencyDistance:
                critical_conflict = conflict
                break
        
        if critical_conflict is None:
            return v_ideal * 0.7, 'forward', 0.0
        
        obs_speed = np.linalg.norm(critical_conflict['obstacle'].velocity)
        if obs_speed < 0.05:
            return v_ideal * 0.3, 'forward', 0.0
        
        reverse_cost, wait_cost = self._calculate_reverse_cost_benefit(
            critical_conflict['obstacle'], critical_conflict['distance']
        )
        
        if reverse_cost < wait_cost - self.cfg.reverseMinBenefit:
            estimated_reverse_dist = max(0.3, self.cfg.safetyDistance - critical_conflict['distance'] + 0.2)
            estimated_reverse_dist = min(estimated_reverse_dist, self.cfg.maxReverseDistance)
            
            self.reverseDecision = 'reverse'
            self.decisionTime = current_time
            self.reverseTargetDistance = estimated_reverse_dist
            self.reverseStartPos = self.pose[:2].copy()
            
            return -self.cfg.reverseVel, 'reverse', estimated_reverse_dist
        else:
            self.reverseDecision = 'wait'
            self.decisionTime = current_time
            return 0.0, 'wait', 0.0
    
    def _calculate_reverse_cost_benefit(self, obstacle: ObstacleState, curr_dist: float) -> Tuple[float, float]:
        """Calculate cost-benefit of reversing vs waiting."""
        estimated_reverse_dist = max(0.3, self.cfg.safetyDistance - curr_dist + 0.2)
        estimated_reverse_dist = min(estimated_reverse_dist, self.cfg.maxReverseDistance)
        
        if not self._is_reverse_path_clear(estimated_reverse_dist):
            return np.inf, 0.0
        
        if is_near_boundary(self.pose, self.cfg.mapSize, self.cfg.mapBoundaryMargin, self.cfg.mapOrigin):
            return np.inf, 0.0
        
        reverse_cost = (estimated_reverse_dist * self.cfg.reverseCostPerMeter + 
                       estimated_reverse_dist / self.cfg.targetLinearVel * self.cfg.waitCostPerSecond)
        
        obs_speed = np.linalg.norm(obstacle.velocity)
        if obs_speed > 0.05:
            clearance_dist = curr_dist + self.cfg.safetyDistance
            estimated_wait_time = clearance_dist / obs_speed
        else:
            estimated_wait_time = 10.0
        
        wait_cost = estimated_wait_time * self.cfg.waitCostPerSecond
        
        return reverse_cost, wait_cost
    
    def _is_reverse_path_clear(self, reverse_dist: float) -> bool:
        """Check if reverse path is clear."""
        reverse_heading = self.pose[2] + np.pi
        
        # Map boundaries in world coordinates
        x_min = self.cfg.mapOrigin[0] + self.cfg.mapBoundaryMargin
        x_max = self.cfg.mapOrigin[0] + self.cfg.mapSize[0] - self.cfg.mapBoundaryMargin
        y_min = self.cfg.mapOrigin[1] + self.cfg.mapBoundaryMargin
        y_max = self.cfg.mapOrigin[1] + self.cfg.mapSize[1] - self.cfg.mapBoundaryMargin
        
        num_checks = 8
        for i in range(1, num_checks + 1):
            t = (i / num_checks) * reverse_dist
            check_x = self.pose[0] + t * np.cos(reverse_heading)
            check_y = self.pose[1] + t * np.sin(reverse_heading)
            check_pose = np.array([check_x, check_y, self.pose[2]])
            
            # Check boundaries
            if check_x < x_min or check_x > x_max:
                return False
            if check_y < y_min or check_y > y_max:
                return False
            
            for obs in self.obstacles.values():
                if check_any_collision(check_pose, obs.pose, self.cfg):
                    return False
                dist = np.linalg.norm(check_pose[:2] - obs.pose[:2])
                if dist < self.cfg.safetyDistance:
                    return False
        
        return True

# === ROBOT STATE MACHINE WITH NAVIGATION ===

class RobotStateMachine:
    def __init__(self, robot_id: str, mqtt_broker: str = "localhost", config: Optional[Config] = None, 
                 enable_visualization: bool = True):
        self.robot_id = robot_id
        self.current_state = RobotState.WAIT
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = mqtt_broker
        
        # Navigation system
        self.config = config if config is not None else Config()
        self.navigator = RobotNavigator(self.config)

        # Visualization
        self.enable_visualization = enable_visualization
        self.visualizer = None
        if enable_visualization:
            self.visualizer = NavigationVisualizer(self.config, robot_id)
            self.visualizer.show()
        
        # MQTT setup
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Control flags
        self.should_exit = False
        self.last_control_time = time.time()
        self.pose_initialized = False  # Track if we've received first MoCap pose
        
        # MoCap interface placeholder (you'll need to implement this)
        self.mocap_interface = None
        
        # MQTT topic for sending velocity commands to Limo
        self.command_topic = f"rb/limo{self.robot_id}/command"
        
        print(f"Robot State Machine initialized for robot {robot_id}")
        print(f"Will publish motor commands to: {self.command_topic}")
        self.last_odom_pose = None

        
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        print(f"Connected to MQTT broker with result code {rc}")
        
        cmd_topic = f"cmd/limo{self.robot_id}"
        goal_topic = f"goal/limo{self.robot_id}"
        odom_topic = f"rb/limo{self.robot_id}/odom"  # Subscribe to odometry
        
        client.subscribe(cmd_topic)
        client.subscribe(goal_topic)
        client.subscribe(odom_topic)
        print(f"Subscribed to {cmd_topic}, {goal_topic}, and {odom_topic}")

    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        if topic.startswith("cmd/"):
            self.handle_command(payload)
        elif topic.startswith("goal/"):
            self.handle_goal(payload)
        elif topic.endswith("/odom"):
            self.handle_odometry(payload)

    
    def handle_odometry(self, odom_json: str):
        """Handle odometry updates from robot"""
        try:
            odom_data = json.loads(odom_json)
            x = odom_data.get('x', 0.0)
            y = odom_data.get('y', 0.0)
            theta = odom_data.get('theta', 0.0)
            
            self.last_odom_pose = np.array([x, y, theta])
            
            # Try to match this to MoCap data
            if self.mocap_interface is not None:
                self.mocap_interface.match_robot_to_mocap(self.last_odom_pose)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing odometry message: {e}")
    
    def update_mocap_data(self):
        """
        Update robot pose and obstacle positions from MoCap system.
        Now uses odometry matching!
        """
        if self.mocap_interface is None:
            return
        
        # Get MY robot's pose from MoCap (already matched via odometry)
        my_pose = self.mocap_interface.get_rigid_body_pose(f"limo{self.robot_id}")
        if my_pose is not None:
            self.navigator.set_pose(my_pose['x'], my_pose['y'], my_pose['theta'])
            
            if not self.pose_initialized:
                self.pose_initialized = True
                print(f"✓ Initial MoCap pose: ({my_pose['x']:.2f}, {my_pose['y']:.2f})")
        
        # Get ALL other robots as obstacles (automatically excludes your robot)
        all_obstacles = self.mocap_interface.get_all_rigid_bodies()
        for obstacle_name, pose in all_obstacles.items():
            self.navigator.update_obstacle(
                obstacle_name,
                pose['x'],
                pose['y'],
                pose.get('theta', 0.0)
            )

    def handle_command(self, command: str):
        """Handle run state commands"""
        print(f"Received command: {command}")
        
        try:
            new_state = RobotState(command)
            self.transition_to(new_state)
        except ValueError:
            print(f"Unknown command: {command}")
    
    def handle_goal(self, goal_json: str):
        """Handle goal position updates"""
        if self.current_state != RobotState.WAIT:
            print(f"Ignoring goal update - not in WAIT state (current: {self.current_state.value})")
            return
        
        try:
            goal_data = json.loads(goal_json)
            x, y = goal_data["goal"]
            self.navigator.set_goal(x, y)
            print(f"Goal updated to: ({x}, {y})")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing goal message: {e}")
    
    def transition_to(self, new_state: RobotState):
        """Handle state transitions with appropriate actions"""
        old_state = self.current_state
        self.current_state = new_state
        
        print(f"State transition: {old_state.value} -> {new_state.value}")
        
        if new_state == RobotState.WAIT:
            self.on_wait()
        elif new_state == RobotState.GO:
            self.on_go()
        elif new_state == RobotState.STOP:
            self.on_stop()
        elif new_state == RobotState.HALT:
            self.on_halt()
    
    def on_wait(self):
        """Actions when entering WAIT state"""
        print("Entering WAIT state - ready to receive goal")
        self.stop_robot()
    
    def on_go(self):
        """Actions when entering GO state"""
        print("Entering GO state - starting navigation")
        if self.navigator.goal is not None:
            print(f"Navigating to goal: {self.navigator.goal}")
        else:
            print("Warning: No goal set!")
    
    def on_stop(self):
        """Actions when entering STOP state - must stop within 100ms"""
        print("Entering STOP state - stopping robot (100ms deadline)")
        self.stop_robot()
    
    def on_halt(self):
        """Actions when entering HALT state - must stop within 100ms and exit"""
        print("Entering HALT state - stopping robot and preparing to exit")
        self.stop_robot()
        self.should_exit = True
    
    def stop_robot(self):
        """Send stop command to robot motors"""
        self.send_velocity_command(0.0, 0.0)
        print("Robot motors stopped")
    
    def send_velocity_command(self, linear_vel: float, angular_vel: float):
        """
        Send velocity command to robot hardware via MQTT.
        
        Args:
            linear_vel: Linear velocity in m/s (v)
            angular_vel: Angular velocity in rad/s (w)
        
        The command is published to rb/limo{robot_id}/command
        Format: {"v": linear_vel, "w": angular_vel}
        
        Note: Make sure the MQTT-to-TCP bridge is running for your Limo!
        """
        # Create command message in the format expected by Limo bridge
        cmd = {
            "v": float(linear_vel),
            "w": float(angular_vel)
        }
        
        # Publish to MQTT topic
        msg_str = json.dumps(cmd)
        self.mqtt_client.publish(self.command_topic, msg_str)
        
        # Optional: log commands (can be commented out for production)
        # print(f"Sent command: v={linear_vel:.3f} m/s, w={angular_vel:.3f} rad/s")
    
    def update_mocap_data(self):
        """
        Update robot pose and obstacle positions from MoCap system.
        
        TODO: Implement your MoCap interface here.
        This should:
        1. Get current robot pose (x, y, theta)
        2. Get positions of all other robots/obstacles
        3. Update navigator with this information
        """
        if self.mocap_interface is None:
            print("Warning: No MoCap interface configured!")
            return
        
        # Get MY robot's pose from MoCap
        my_pose = self.mocap_interface.get_rigid_body_pose(f"limo{self.robot_id}")
        if my_pose is not None:
            self.navigator.set_pose(my_pose['x'], my_pose['y'], my_pose['theta'])
            
            # Mark that we've received at least one pose
            if not self.pose_initialized:
                self.pose_initialized = True
                print(f"Initial pose from MoCap: ({my_pose['x']:.2f}, {my_pose['y']:.2f}, {np.rad2deg(my_pose['theta']):.1f}°)")
        else:
            print(f"Warning: Could not get pose for limo{self.robot_id} from MoCap")
        
        # Get ALL other robots and treat them as obstacles
        all_poses = self.mocap_interface.get_all_rigid_bodies()
        for rigid_body_id, pose in all_poses.items():
            if rigid_body_id != f"limo{self.robot_id}":  # Skip my own robot
                self.navigator.update_obstacle(
                    rigid_body_id,
                    pose['x'],
                    pose['y'],
                    pose.get('theta', 0.0)
                )
    
    def run(self):
        """Main control loop"""
        # Connect to MQTT broker
        self.mqtt_client.connect(self.mqtt_broker, 1883, 60)
        self.mqtt_client.loop_start()
        
        print("Robot state machine running...")
        print(f"Control rate: {self.config.controlRate} Hz")
        
        try:
            while not self.should_exit:
                loop_start = time.time()
                
                # Update MoCap data
                self.update_mocap_data()
                
                # Execute state-specific behavior
                if self.current_state == RobotState.GO:
                    self.navigate_step()
                elif self.current_state == RobotState.WAIT:
                    # Idle - robot is stopped
                    pass
                elif self.current_state == RobotState.STOP:
                    # Robot is stopped but ready to resume
                    pass
                elif self.current_state == RobotState.HALT:
                    # Should exit - this is handled by should_exit flag
                    pass
                
                # Maintain control rate
                elapsed = time.time() - loop_start
                sleep_time = self.config.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        finally:
            self.cleanup()
    
    def navigate_step(self):
        """Execute one step of navigation - ONLY when in GO state"""
        if self.current_state != RobotState.GO:
            return
        
        # Safety check: make sure we have pose data from MoCap
        if not self.pose_initialized:
            print("Warning: Waiting for MoCap pose data before navigating...")
            return
        
        # Safety check: make sure we have a goal
        if self.navigator.goal is None:
            print("Warning: No goal set, cannot navigate!")
            self.transition_to(RobotState.WAIT)
            return
        
        # Compute control commands
        lin_vel, ang_vel, status = self.navigator.compute_control()
        
        # Send commands to robot
        self.send_velocity_command(lin_vel, ang_vel)
        
        # Check if goal reached
        if status.get('goal_reached', False):
            print("✓ Goal reached! Transitioning to WAIT state")
            self.transition_to(RobotState.WAIT)
        
        # Log warnings
        if status.get('stuck', False):
            print(f"Warning: Robot appears stuck (counter: {self.navigator.stuckCounter})")
        
        if status.get('reversing', False):
            print(f"Info: Robot reversing to avoid obstacle")
        
        # Periodic status update
        current_time = time.time()
        if current_time - self.last_control_time > 1.0:  # Every 1 second
            print(f"Status: dist_to_goal={status.get('distance_to_goal', 0):.2f}m, "
                  f"obstacles={status.get('num_obstacles', 0)}, "
                  f"v={lin_vel:.2f}m/s, ω={np.rad2deg(ang_vel):.1f}°/s")
            self.last_control_time = current_time
    
    def cleanup(self):
        """Cleanup resources before exit"""
        print("Cleaning up...")
        self.stop_robot()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        print("Exited gracefully")


# === OPTITRACK MOCAP INTERFACE ===

class OptiTrackMoCapInterface:
    """
    OptiTrack interface using NatNet protocol.
    Receives rigid body poses in real-time.
    """
    
    def __init__(self, server_ip: str = "239.255.42.99", data_port: int = 1511):
        """
        Args:
            server_ip: Multicast address for OptiTrack data stream
            data_port: Port for data stream (default 1511)
        """
        self.server_ip = server_ip
        self.data_port = data_port
        
        # Storage for ALL rigid body poses
        self.all_rigid_bodies = {}  # {rb_id: {'x', 'y', 'theta', 'timestamp'}}
        self.lock = Lock()

        self.my_rb_id = None
        
        # Networking
        self.socket = None
        self.running = False
        self.thread = None
        
        print(f"OptiTrack interface initialized (multicast: {server_ip}:{data_port})")
    
    def start(self):
        """Start receiving MoCap data"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('', self.data_port))
        
        mreq = struct.pack("4sl", socket.inet_aton(self.server_ip), socket.INADDR_ANY)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        self.running = True
        self.thread = Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        
        print("✓ OptiTrack streaming started")
    
    def stop(self):
        """Stop receiving MoCap data"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join()
        print("OptiTrack streaming stopped")

    
    def _receive_loop(self):
        """Receive and parse NatNet packets"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65535)
                self._parse_frame(data)
            except Exception as e:
                if self.running:
                    print(f"Error receiving MoCap data: {e}")

    
    def _parse_frame(self, data):
        """Parse NatNet frame data and store ALL rigid bodies"""
        try:
            offset = 0
            
            # Message ID (2 bytes)
            message_id = struct.unpack('H', data[offset:offset+2])[0]
            offset += 2
            
            # Packet size (2 bytes)
            packet_size = struct.unpack('H', data[offset:offset+2])[0]
            offset += 2
            
            # Only process frame data (message_id == 7)
            if message_id != 7:
                return
            
            # Frame number (4 bytes)
            frame_num = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            # Number of rigid bodies (4 bytes)
            num_rigid_bodies = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            
            # Parse each rigid body
            for i in range(num_rigid_bodies):
                # Rigid body ID (4 bytes)
                rb_id = struct.unpack('I', data[offset:offset+4])[0]
                offset += 4
                
                # Position (12 bytes: 3 floats for x, y, z)
                x, y, z = struct.unpack('fff', data[offset:offset+12])
                offset += 12
                
                # Orientation quaternion (16 bytes: 4 floats for qx, qy, qz, qw)
                qx, qy, qz, qw = struct.unpack('ffff', data[offset:offset+16])
                offset += 16
                
                # Convert to 2D pose
                pose_x = x
                pose_y = z  # Swap Y and Z for 2D navigation
                
                # Calculate yaw from quaternion
                siny_cosp = 2.0 * (qw * qy + qz * qx)
                cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
                theta = np.arctan2(siny_cosp, cosy_cosp)
                
                # Store ALL rigid bodies by their OptiTrack ID
                with self.lock:
                    self.all_rigid_bodies[rb_id] = {
                        'x': pose_x,
                        'y': pose_y,
                        'theta': theta,
                        'timestamp': time.time()
                    }
                
        except Exception as e:
            # Silently ignore parse errors
            pass
    
    def match_robot_to_mocap(self, robot_odom_pose: np.ndarray, max_distance: float = 0.5):
        """
        Match your robot's odometry to a MoCap rigid body.
        Call this once you have odometry data.
        
        Args:
            robot_odom_pose: [x, y, theta] from robot's odometry
            max_distance: Maximum distance to consider a match (meters)
        """
        with self.lock:
            if len(self.all_rigid_bodies) == 0:
                return False
            
            # Find closest rigid body to odometry position
            robot_pos = robot_odom_pose[:2]
            closest_id = None
            closest_dist = float('inf')
            
            for rb_id, pose in self.all_rigid_bodies.items():
                rb_pos = np.array([pose['x'], pose['y']])
                dist = np.linalg.norm(rb_pos - robot_pos)
                
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = rb_id
            
            # Only match if within reasonable distance
            if closest_dist < max_distance:
                if self.my_rb_id != closest_id:
                    self.my_rb_id = closest_id
                    print(f"✓ Matched robot odometry to MoCap rigid body ID={closest_id} (dist={closest_dist:.2f}m)")
                return True
            else:
                print(f"⚠ No MoCap rigid body within {max_distance}m of odometry position")
                return False

    

    def get_rigid_body_pose(self, rigid_body_name: str) -> Optional[Dict]:
        """
        Get pose for your robot (must call match_robot_to_mocap first).
        """
        if "limo" in rigid_body_name:
            # This is asking for OUR robot
            with self.lock:
                if self.my_rb_id is None:
                    return None  # Haven't matched yet
                
                if self.my_rb_id in self.all_rigid_bodies:
                    pose = self.all_rigid_bodies[self.my_rb_id]
                    age = time.time() - pose['timestamp']
                    
                    if age < 0.5:  # Fresh data
                        return {
                            'x': pose['x'],
                            'y': pose['y'],
                            'theta': pose['theta']
                        }
        
        return None

    
    def get_all_rigid_bodies(self) -> Dict[str, Dict]:
        """
        Get all rigid bodies EXCEPT your robot (i.e., obstacles only).
        """
        result = {}
        current_time = time.time()
        
        with self.lock:
            for rb_id, pose in self.all_rigid_bodies.items():
                # Skip YOUR robot - only return obstacles
                if rb_id == self.my_rb_id:
                    continue
                
                age = current_time - pose['timestamp']
                if age < 0.5:  # Only recent data
                    # Name obstacles by their OptiTrack ID
                    result[f"obstacle_{rb_id}"] = {
                        'x': pose['x'],
                        'y': pose['y'],
                        'theta': pose['theta']
                    }
        
        return result

# === VISUALIZATION ===

class NavigationVisualizer:
    """Real-time visualization of robot navigation."""
    
    def __init__(self, config: Config, robot_id: str):
        self.cfg = config
        self.robot_id = robot_id
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal')
        
        # Set up plot limits based on map
        margin = 0.5
        self.ax.set_xlim(config.mapOrigin[0] - margin, 
                         config.mapOrigin[0] + config.mapSize[0] + margin)
        self.ax.set_ylim(config.mapOrigin[1] - margin, 
                         config.mapOrigin[1] + config.mapSize[1] + margin)
        
        self.ax.set_xlabel('X (m) - toward hallway', fontsize=12)
        self.ax.set_ylabel('Y (m) - toward street', fontsize=12)
        self.ax.set_title(f'Robot limo{robot_id} Navigation', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Draw map boundaries
        self._draw_map_boundaries()
        
        # Initialize plot elements (will be updated)
        self.robot_circle = None
        self.robot_arrow = None
        self.goal_marker = None
        self.path_line = None
        self.trajectory_line = None
        self.obstacle_circles = []
        self.obstacle_arrows = []
        self.obstacle_labels = []
        
        # Text status
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                       verticalalignment='top', fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
    def _draw_map_boundaries(self):
        """Draw map boundaries and safe zone"""
        # Outer boundary (map limits)
        map_rect = Rectangle(
            (self.cfg.mapOrigin[0], self.cfg.mapOrigin[1]),
            self.cfg.mapSize[0], self.cfg.mapSize[1],
            fill=False, edgecolor='black', linewidth=2, linestyle='--',
            label='Map boundary'
        )
        self.ax.add_patch(map_rect)
        
        # Safe zone (with margin)
        safe_rect = Rectangle(
            (self.cfg.mapOrigin[0] + self.cfg.mapBoundaryMargin, 
             self.cfg.mapOrigin[1] + self.cfg.mapBoundaryMargin),
            self.cfg.mapSize[0] - 2*self.cfg.mapBoundaryMargin,
            self.cfg.mapSize[1] - 2*self.cfg.mapBoundaryMargin,
            fill=False, edgecolor='red', linewidth=1, linestyle=':',
            label='Safe zone'
        )
        self.ax.add_patch(safe_rect)
        
        # Origin marker
        self.ax.plot(0, 0, 'k+', markersize=15, markeredgewidth=2, label='Origin (0,0)')
        
        self.ax.legend(loc='upper right', fontsize=9)
    
    def update(self, navigator: RobotNavigator, state: RobotState):
        """Update visualization with current navigation state."""
        
        # Clear previous dynamic elements
        if self.robot_circle:
            self.robot_circle.remove()
        if self.robot_arrow:
            self.robot_arrow.remove()
        if self.goal_marker:
            self.goal_marker.remove()
        if self.path_line:
            self.path_line.remove()
        if self.trajectory_line:
            self.trajectory_line.remove()
        
        for circle in self.obstacle_circles:
            circle.remove()
        for arrow in self.obstacle_arrows:
            arrow.remove()
        for label in self.obstacle_labels:
            label.remove()
        
        self.obstacle_circles.clear()
        self.obstacle_arrows.clear()
        self.obstacle_labels.clear()
        
        # Draw robot
        self._draw_robot(navigator.pose, state)
        
        # Draw goal
        if navigator.goal is not None:
            self._draw_goal(navigator.goal)
        
        # Draw planned path
        if navigator.path is not None:
            self._draw_path(navigator.path)
        
        # Draw trajectory history
        if len(navigator.trajectory) > 1:
            self._draw_trajectory(navigator.trajectory)
        
        # Draw obstacles
        self._draw_obstacles(navigator.obstacles)
        
        # Update status text
        self._update_status_text(navigator, state)
        
        # Refresh plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _draw_robot(self, pose: np.ndarray, state: RobotState):
        """Draw robot as circle + heading arrow"""
        x, y, theta = pose
        
        # Color based on state
        state_colors = {
            RobotState.WAIT: 'gray',
            RobotState.GO: 'green',
            RobotState.STOP: 'orange',
            RobotState.HALT: 'red'
        }
        color = state_colors.get(state, 'blue')
        
        # Robot body (circle)
        self.robot_circle = Circle((x, y), self.cfg.robotClearanceRadius,
                                   color=color, alpha=0.6, linewidth=2,
                                   edgecolor='black', zorder=5)
        self.ax.add_patch(self.robot_circle)
        
        # Heading arrow
        arrow_length = 0.3
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        
        self.robot_arrow = FancyArrow(x, y, dx, dy,
                                      width=0.1, head_width=0.2, head_length=0.1,
                                      color='black', zorder=6)
        self.ax.add_patch(self.robot_arrow)
    
    def _draw_goal(self, goal: np.ndarray):
        """Draw goal position"""
        self.goal_marker, = self.ax.plot(goal[0], goal[1], 'r*', 
                                         markersize=20, markeredgewidth=2,
                                         markeredgecolor='black', zorder=4,
                                         label='Goal')
    
    def _draw_path(self, path: Path):
        """Draw planned path"""
        self.path_line, = self.ax.plot(path.x, path.y, 'b--', 
                                       linewidth=2, alpha=0.7, 
                                       label='Planned path', zorder=2)
    
    def _draw_trajectory(self, trajectory: list):
        """Draw robot's trajectory history"""
        traj_array = np.array(trajectory)
        self.trajectory_line, = self.ax.plot(traj_array[:, 0], traj_array[:, 1],
                                            'c-', linewidth=1.5, alpha=0.5,
                                            label='Trajectory', zorder=1)
    
    def _draw_obstacles(self, obstacles: Dict[str, ObstacleState]):
        """Draw dynamic obstacles (other robots)"""
        for obs_id, obs in obstacles.items():
            x, y, theta = obs.pose
            
            # Obstacle body (circle)
            circle = Circle((x, y), self.cfg.robotClearanceRadius,
                          color='purple', alpha=0.4, linewidth=1.5,
                          edgecolor='purple', linestyle='--', zorder=3)
            self.ax.add_patch(circle)
            self.obstacle_circles.append(circle)
            
            # Heading arrow
            arrow_length = 0.25
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            
            arrow = FancyArrow(x, y, dx, dy,
                              width=0.08, head_width=0.15, head_length=0.08,
                              color='purple', alpha=0.6, zorder=3)
            self.ax.add_patch(arrow)
            self.obstacle_arrows.append(arrow)
            
            # Velocity vector (if moving)
            speed = np.linalg.norm(obs.velocity)
            if speed > 0.05:
                vel_scale = 0.5
                dvx = obs.velocity[0] * vel_scale
                dvy = obs.velocity[1] * vel_scale
                vel_arrow = FancyArrow(x, y, dvx, dvy,
                                      width=0.05, head_width=0.1, head_length=0.05,
                                      color='red', alpha=0.7, zorder=3)
                self.ax.add_patch(vel_arrow)
                self.obstacle_arrows.append(vel_arrow)
            
            # Label
            label = self.ax.text(x, y + self.cfg.robotClearanceRadius + 0.15,
                               obs_id, ha='center', va='bottom',
                               fontsize=8, color='purple', fontweight='bold',
                               zorder=4)
            self.obstacle_labels.append(label)
    
    def _update_status_text(self, navigator: RobotNavigator, state: RobotState):
        """Update status text display"""
        status_lines = [
            f"State: {state.value}",
            f"Position: ({navigator.pose[0]:.2f}, {navigator.pose[1]:.2f})",
            f"Heading: {np.rad2deg(navigator.pose[2]):.1f}°"
        ]
        
        if navigator.goal is not None:
            dist = np.linalg.norm(navigator.pose[:2] - navigator.goal)
            status_lines.append(f"Distance to goal: {dist:.2f}m")
        
        status_lines.append(f"Obstacles tracked: {len(navigator.obstacles)}")
        status_lines.append(f"Replans: {navigator.replanCount}")
        
        if navigator.stuckCounter > 0:
            status_lines.append(f"⚠ Stuck counter: {navigator.stuckCounter}")
        
        self.status_text.set_text('\n'.join(status_lines))
    
    def show(self):
        """Show the plot window"""
        plt.ion()  # Interactive mode
        plt.show()
    
    def close(self):
        """Close the plot window"""
        plt.close(self.fig)

# === MAIN ===

if __name__ == "__main__":
    # Configuration
    ROBOT_ID = "777"  # change based on robot IP (777, 805, 809, etc.)
    MQTT_BROKER = "rasticvm.lan"
    
    # OptiTrack Configuration
    OPTITRACK_SERVER_IP = "239.255.42.99"  # Multicast address (standard for OptiTrack)
    OPTITRACK_DATA_PORT = 1511  # Standard NatNet data port
    
    # Create custom config if needed
    config = Config()
    config.controlRate = 30.0  # 30 Hz
    config.mapSize = np.array([9.0, 5.0])  # 9m (width) x 5m (depth)
    config.mapOrigin = np.array([-4.5, -2.5])  # Origin at center of room
    config.targetLinearVel = 0.3  # 0.3 m/s
    config.maxAngVel = np.deg2rad(90)  # 90 deg/s
    
    print("\n=== ARENA MoCap Room Configuration ===")
    print(f"Map size: {config.mapSize[0]}m x {config.mapSize[1]}m")
    print(f"X range: [{config.mapOrigin[0]:.1f}, {config.mapOrigin[0] + config.mapSize[0]:.1f}]m (toward hallway windows)")
    print(f"Y range: [{config.mapOrigin[1]:.1f}, {config.mapOrigin[1] + config.mapSize[1]:.1f}]m (toward street)")
    print(f"Origin: (0, 0) at center red tape rectangle")
    print(f"Example goals:")
    print(f"  (4.5, 2.5) = near hallway door")
    print(f"  (-4.5, -2.5) = near RASTIC room door")
    print("="*50)
    print("\n=== MQTT Topics ===")
    print(f"Listening for commands on: cmd/limo{ROBOT_ID}")
    print(f"Listening for goals on: goal/limo{ROBOT_ID}")
    print(f"Publishing motor commands to: rb/limo{ROBOT_ID}/command")
    print("="*50)
    print("\n=== Required Components ===")
    print("1. MQTT Broker running at rasticvm.lan")
    print(f"2. MQTT-to-TCP bridge running for limo{ROBOT_ID}")
    print("   (Use your existing bridge script to connect MQTT to Limo TCP)")
    print("3. OptiTrack MoCap system streaming data")
    print(f"   Server: {OPTITRACK_SERVER_IP}:{OPTITRACK_DATA_PORT}")
    print("="*50 + "\n")
    
    # Create state machine with visualization
    robot = RobotStateMachine(ROBOT_ID, MQTT_BROKER, config, enable_visualization=True)
    
    # Set up OptiTrack MoCap interface
    print("Setting up OptiTrack interface...")
    mocap = OptiTrackMoCapInterface(
        server_ip=OPTITRACK_SERVER_IP,
        data_port=OPTITRACK_DATA_PORT
    )
    mocap.start()
    robot.mocap_interface = mocap
    
    # Wait for first pose data
    print(f"\n✓ Ready! Waiting for odometry from limo{ROBOT_ID} to match with MoCap...")
    print("  Robot will auto-match itself to MoCap rigid bodies using odometry.")
    print("  All other rigid bodies will be treated as obstacles.\n")
    
    # Run
    try:
        robot.run()
    finally:
        mocap.stop()
