import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, List, Optional, Tuple

# === CONFIGURATION AND HELPER STRUCTURES (No change here) ===

class Config:
    """Controller and Planning Configuration Parameters."""
    def __init__(self):
        # Pure Pursuit Controller
        self.lookaheadDist: float = 0.35
        self.targetLinearVel: float = 0.25
        self.minTurnRadius: float = 0.2
        self.recoveryDuration: float = 1.0
        self.posTolerance: float = 0.2
        self.headingTolerance: float = np.deg2rad(25)
        self.maxLinVel: float = 0.25
        self.maxAngVel: float = np.deg2rad(60)

        # 🤖 PHYSICAL ROBOT DIMENSIONS (40 cm x 30 cm)
        self.robotWidth: float = 0.40  # 40 cm (transverse)
        self.robotLength: float = 0.30  # 30 cm (longitudinal/heading)
        self.safetyBuffer: float = 0.1 # Additional clearance buffer

        # Calculate the clearance radius (radius of the circle that circumscribes the box + buffer)
        R_box_diagonal = np.sqrt(self.robotWidth**2 + self.robotLength**2) / 2.0
        self.robotClearanceRadius: float = R_box_diagonal + self.safetyBuffer
        
        # Path planning parameters (RRT*)
        self.replanInterval: float = 1.0
        self.rrtMaxIter: int = 500
        self.rrtStepSize: float = 0.3
        self.rrtGoalBias: float = 0.15
        self.rrtNeighborRadius: float = 0.8

        # Dynamic obstacle parameters (was CFG.robotSafetyRadius)
        self.predictionHorizon: float = 3.0
        self.velocityScaleFactor: float = 1.5

        # Kalman filter parameters
        self.kf_processNoise: float = 0.1
        self.kf_measureNoise: float = 0.05

class SimParams:
    """Simulation Parameters."""
    def __init__(self):
        self.numRobots: int = 12
        self.maxTime: float = 300.0  # Maximum simulation time [s]
        self.dt: float = 1.0 / 30.0  # Time step [s] (30 Hz control rate)
        self.mapSize: np.ndarray = np.array([5.5, 5.0])  # [width, height] in meters
        self.realTimeMode: bool = True  # Set to true for real-time visualization

class Path:
    """Represents a robot's planned path."""
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.s: np.ndarray = self._calculate_arc_length(x, y)

    @staticmethod
    def _calculate_arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.insert(np.cumsum(ds), 0, 0.0)
        return s

class RobotState:
    """Represents a single robot's state and internal variables."""
    def __init__(self, initial_pose: np.ndarray, goal: np.ndarray, initial_path: Path, cfg: Config):
        self.pose: np.ndarray = initial_pose  # [x, y, theta]
        self.velocity: np.ndarray = np.array([0.0, 0.0])  # [vx, vy]
        self.goal: np.ndarray = goal  # [gx, gy]
        self.trajectory: List[np.ndarray] = [initial_pose[:2].copy()]
        self.reached: bool = False
        self.reachedTime: float = np.inf
        self.lastReplanTime: float = -np.inf
        self.replanCount: int = 0
        self.recoveryTimer: Optional[float] = None
        self.errorHistory: List[float] = []
        self.path: Path = initial_path

        # Kalman filter tracking state [x, y, vx, vy]
        self.kf_state: np.ndarray = np.array([initial_pose[0], initial_pose[1], 0.0, 0.0])
        self.kf_P: np.ndarray = np.eye(4) * 0.1
        Q = np.eye(4) * cfg.kf_processNoise
        Q[:2, :2] = Q[:2, :2] * 0.1
        self.kf_Q: np.ndarray = Q
        self.kf_R: np.ndarray = np.eye(2) * cfg.kf_measureNoise

class RRTNode:
    """Node structure for the RRT* search tree."""
    def __init__(self, x: float, y: float, parent: int, cost: float):
        self.x: float = x
        self.y: float = y
        self.parent: int = parent
        self.cost: float = cost

class Obstacle:
    """Represents a circular obstacle (static or dynamic prediction)."""
    def __init__(self, x: float, y: float, radius: float):
        self.x: float = x
        self.y: float = y
        self.radius: float = radius

# === HELPER FUNCTIONS (No change here) ===

def wrap_to_pi(angle: float) -> float:
    """Wraps an angle to the range (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def create_other_robots_list(robots: Dict[str, RobotState], my_robot_id: str) -> List[Dict[str, float]]:
    """Creates a list of other robots' Kalman-filtered states for path planning."""
    other_robots_list = []
    for rid, robot in robots.items():
        if rid == my_robot_id or robot.reached:
            continue
        
        other_robots_list.append({
            'x': robot.kf_state[0],
            'y': robot.kf_state[1],
            'vx': robot.kf_state[2],
            'vy': robot.kf_state[3]
        })
    return other_robots_list

def create_dynamic_obstacles_sim(other_robots_list: List[Dict[str, float]], cfg: Config) -> List[Obstacle]:
    """Creates a list of dynamic obstacles based on other robots' predicted trajectories."""
    obstacles: List[Obstacle] = []
    # Static obstacles are an empty list in this simulator, so we skip that part.

    # Add dynamic robot obstacles
    for robot in other_robots_list:
        x, y, vx, vy = robot['x'], robot['y'], robot['vx'], robot['vy']
        speed = np.sqrt(vx**2 + vy**2)
        
        # Current position: Use the clearance radius
        obstacles.append(Obstacle(x, y, cfg.robotClearanceRadius))
        
        # Predicted positions (if moving)
        if speed > 0.02:
            num_predictions = 5
            for j in range(1, num_predictions + 1):
                pred_time = (j / num_predictions) * cfg.predictionHorizon
                
                pred_x = x + vx * pred_time
                pred_y = y + vy * pred_time
                
                # Radius increases with prediction time to account for uncertainty
                pred_radius = cfg.robotClearanceRadius + speed * cfg.velocityScaleFactor * (j / num_predictions)
                obstacles.append(Obstacle(pred_x, pred_y, pred_radius))
                
    return obstacles

def is_collision_free_sim(x1: float, y1: float, x2: float, y2: float, obstacles: List[Obstacle]) -> bool:
    """Checks if the line segment (x1,y1) to (x2,y2) is free of obstacles."""
    num_checks = 10
    
    for i in range(num_checks + 1):
        t = i / num_checks
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        for obstacle in obstacles:
            dist = np.sqrt((x - obstacle.x)**2 + (y - obstacle.y)**2)
            if dist < obstacle.radius:
                return False
    return True

def extract_path_sim(tree: List[RRTNode], goal_idx: int) -> Path:
    """Reconstructs the path from the RRT* tree."""
    pathX, pathY = [], []
    current_idx = goal_idx
    
    while current_idx != 0:
        node = tree[current_idx - 1] # -1 because tree is 1-indexed in MATLAB, 0-indexed in Python list
        pathX.insert(0, node.x)
        pathY.insert(0, node.y)
        current_idx = node.parent
        
    return Path(np.array(pathX), np.array(pathY))

def rrt_star_planner(start_pose: np.ndarray, goal_pos: np.ndarray, obstacles: List[Obstacle], cfg: Config, map_size: np.ndarray) -> Optional[Path]:
    """Rapidly-exploring Random Tree Star (RRT*) path planner."""
    # MATLAB tree is 1-indexed. Python list will be 0-indexed, but parent indices will be 1-based (0 means no parent).
    tree: List[RRTNode] = [RRTNode(start_pose[0], start_pose[1], 0, 0.0)]
    
    goalX, goalY = goal_pos[0], goal_pos[1]
    xMin, xMax = 0.0, map_size[0]
    yMin, yMax = 0.0, map_size[1]
    
    for _ in range(cfg.rrtMaxIter):
        # 1. Sample point
        if np.random.rand() < cfg.rrtGoalBias:
            xRand, yRand = goalX, goalY
        else:
            xRand = xMin + np.random.rand() * (xMax - xMin)
            yRand = yMin + np.random.rand() * (yMax - yMin)

        # 2. Find Nearest
        distances = np.array([np.sqrt((node.x - xRand)**2 + (node.y - yRand)**2) for node in tree])
        nearestIdx_py = np.argmin(distances)
        nearestIdx_matlab = nearestIdx_py + 1 # Convert to 1-based index
        xNearest = tree[nearestIdx_py].x
        yNearest = tree[nearestIdx_py].y

        # 3. Steer
        angle = np.arctan2(yRand - yNearest, xRand - xNearest)
        xNew = xNearest + cfg.rrtStepSize * np.cos(angle)
        yNew = yNearest + cfg.rrtStepSize * np.sin(angle)
        
        # Check map bounds
        if not (xMin <= xNew <= xMax and yMin <= yNew <= yMax):
            continue

        # 4. Check Collision
        if not is_collision_free_sim(xNearest, yNearest, xNew, yNew, obstacles):
            continue
            
        # 5. Find Near Neighbors and Choose Best Parent
        new_node_cost = np.inf
        best_parent_idx_matlab = nearestIdx_matlab
        
        all_dists_new = np.array([np.sqrt((node.x - xNew)**2 + (node.y - yNew)**2) for node in tree])
        nearInds_py = np.where(all_dists_new < cfg.rrtNeighborRadius)[0]

        for idx_py in nearInds_py:
            node = tree[idx_py]
            # Cost to reach new node from this neighbor
            dist_to_new = np.sqrt((node.x - xNew)**2 + (node.y - yNew)**2)
            cost = node.cost + dist_to_new
            
            if cost < new_node_cost and is_collision_free_sim(node.x, node.y, xNew, yNew, obstacles):
                new_node_cost = cost
                best_parent_idx_matlab = idx_py + 1
                
        # 6. Insert New Node
        new_node = RRTNode(xNew, yNew, best_parent_idx_matlab, new_node_cost)
        tree.append(new_node)
        newIdx_py = len(tree) - 1
        newIdx_matlab = newIdx_py + 1
        
        # 7. Rewire Neighbors
        for idx_py in nearInds_py:
            node = tree[idx_py]
            # Cost to reach neighbor from new node
            dist_from_new = np.sqrt((node.x - xNew)**2 + (node.y - yNew)**2)
            cost = new_node_cost + dist_from_new
            
            if cost < node.cost and is_collision_free_sim(xNew, yNew, node.x, node.y, obstacles):
                node.parent = newIdx_matlab
                node.cost = cost
        
        # 8. Check Goal Condition
        distToGoal = np.sqrt((xNew - goalX)**2 + (yNew - goalY)**2)
        if distToGoal < cfg.rrtStepSize:
            path = extract_path_sim(tree, newIdx_matlab)
            return path
            
    # Final check for closest node to goal if RRT*MaxIter is reached
    distances_to_goal = np.array([np.sqrt((node.x - goalX)**2 + (node.y - goalY)**2) for node in tree])
    closestIdx_py = np.argmin(distances_to_goal)
    minDist = distances_to_goal[closestIdx_py]
    
    if minDist < 1.0: # Arbitrary fallback distance
        return extract_path_sim(tree, closestIdx_py + 1)
    else:
        return None

def smooth_path_sim(rough_path: Path) -> Path:
    """
    Path smoothing approximation (replaces Dubins path for portability). 
    Simple straight-line connection between waypoints.
    """
    if len(rough_path.x) < 3:
        return rough_path

    # Select a few waypoints
    num_waypoints = min(len(rough_path.x), 8)
    indices = np.round(np.linspace(0, len(rough_path.x) - 1, num_waypoints)).astype(int)
    wpX = rough_path.x[indices]
    wpY = rough_path.y[indices]
    
    full_path_x, full_path_y = [], []
    
    # Simple straight-line connection between waypoints
    for i in range(num_waypoints - 1):
        x1, y1 = wpX[i], wpY[i]
        x2, y2 = wpX[i+1], wpY[i+1]
        
        nPts = 20
        x_seg = np.linspace(x1, x2, nPts)
        y_seg = np.linspace(y1, y2, nPts)
        
        if i > 0:
            # Skip the first point of subsequent segments to avoid duplicates
            x_seg = x_seg[1:]
            y_seg = y_seg[1:]
            
        full_path_x.extend(x_seg)
        full_path_y.extend(y_seg)

    if not full_path_x:
        return rough_path
        
    return Path(np.array(full_path_x), np.array(full_path_y))

def find_lookahead_sim(robotX: float, robotY: float, planned_path: Path, lookahead_distance: float) -> Tuple[float, float, int, float]:
    """Finds the lookahead point on the path using the Pure Pursuit method."""
    
    # Distance from robot to every point on the path
    all_distances = np.sqrt((planned_path.x - robotX)**2 + (planned_path.y - robotY)**2)
    
    # Closest point on path to robot
    closest_index = np.argmin(all_distances)
    cross_track_error = all_distances[closest_index]
    
    # Target arc length
    sClosest = planned_path.s[closest_index]
    sLookaheadTarget = sClosest + lookahead_distance
    
    # Find the point on the path that corresponds to sLookaheadTarget
    if sLookaheadTarget >= planned_path.s[-1]:
        # If target is beyond the end, use the last point
        lookahead_index = len(planned_path.x) - 1
    else:
        # Find the first index where arc length is >= target
        lookahead_index = np.argmax(planned_path.s >= sLookaheadTarget)
        if lookahead_index == 0 and planned_path.s[0] < sLookaheadTarget: 
             lookahead_index = len(planned_path.x) - 1
    
    lookaheadX = planned_path.x[lookahead_index]
    lookaheadY = planned_path.y[lookahead_index]
    
    return lookaheadX, lookaheadY, lookahead_index, cross_track_error

# === HELPER FUNCTION TO CALCULATE BOUDING BOX CORNERS ===
def get_bounding_box_corners(x_c: float, y_c: float, theta: float, length: float, width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the (x, y) coordinates of the four corners of the oriented bounding box."""
    # Half dimensions
    half_L = width / 2.0
    half_W = length / 2.0
    
    # Coordinates of the four corners in the robot's local frame (relative to center)
    # The 40cm side (width) is transverse, 30cm side (length) is parallel to heading.
    local_corners = np.array([
        [ half_L,  half_W],  # Front-Right
        [ half_L, -half_W],  # Front-Left
        [-half_L, -half_W],  # Back-Left
        [-half_L,  half_W],  # Back-Right
        [ half_L,  half_W]   # Close the loop
    ]).T # Transpose to get a (2, 5) array

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Rotate and translate to global coordinates
    global_corners = R @ local_corners
    
    corner_x = global_corners[0, :] + x_c
    corner_y = global_corners[1, :] + y_c
    
    return corner_x, corner_y


# === MAIN SIMULATOR FUNCTION ===

def multi_robot_simulator():
    """Simulates 12 robots using dynamic path planning."""
    
    np.random.seed(42)  # Set random seed for reproducibility
    
    SIM = SimParams()
    CFG = Config()
    
    static_obstacles: List[Any] = []  # Empty for open space navigation

    # --- INITIALIZE ROBOTS (No Change) ---
    # Calculate required separation based on new bounding box
    min_separation = 2 * CFG.robotClearanceRadius + 0.1 # Minimum center-to-center distance + buffer
    
    print(f'Initializing {SIM.numRobots} robots in open space (min separation: {min_separation:.2f}m)...')
    robots: Dict[str, RobotState] = {}
    edgeMargin = 0.5
    
    for i in range(1, SIM.numRobots + 1):
        robotId = f'robot{i:02d}'
        
        # Generate random start position (with minimum separation)
        valid_start = False
        attempts = 0
        x, y = 0.0, 0.0
        while not valid_start and attempts < 100:
            x = edgeMargin + np.random.rand() * (SIM.mapSize[0] - 2 * edgeMargin)
            y = edgeMargin + np.random.rand() * (SIM.mapSize[1] - 2 * edgeMargin)
            
            valid_start = True
            for robot in robots.values():
                dist = np.sqrt((x - robot.pose[0])**2 + (y - robot.pose[1])**2)
                if dist < min_separation:
                    valid_start = False
                    break
            attempts += 1
            
        # Generate random goal position (far from start)
        valid_goal = False
        attempts = 0
        gx, gy = 0.0, 0.0
        while not valid_goal and attempts < 100:
            gx = edgeMargin + np.random.rand() * (SIM.mapSize[0] - 2 * edgeMargin)
            gy = edgeMargin + np.random.rand() * (SIM.mapSize[1] - 2 * edgeMargin)
            
            dist_from_start = np.sqrt((gx - x)**2 + (gy - y)**2)
            if dist_from_start > 2.0:
                valid_goal = True
            attempts += 1
            
        # Initialize robot state
        theta = np.random.rand() * 2 * np.pi
        initial_pose = np.array([x, y, theta])
        goal_pos = np.array([gx, gy])
        
        # Initial straight-line path
        nPts = 20
        path_x = np.linspace(x, gx, nPts)
        path_y = np.linspace(y, gy, nPts)
        initial_path = Path(path_x, path_y)
        
        robots[robotId] = RobotState(initial_pose, goal_pos, initial_path, CFG)
        
        print(f'  {robotId}: Start({x:.2f}, {y:.2f}) -> Goal({gx:.2f}, {gy:.2f})')

    # --- VISUALIZATION SETUP (Matplotlib) ---
    if SIM.realTimeMode:
        plt.ion() # Turn on interactive mode
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Multi-Robot Simulation - Real-Time Path Planning (Open Space)')
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_xlim([-0.5, SIM.mapSize[0]])
    ax.set_ylim([-0.5, SIM.mapSize[1]])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    
    # Suppress the MatplotlibDeprecationWarning
    try:
        colors = plt.cm.get_cmap('hsv', SIM.numRobots)
    except Exception:
        colors = plt.get_cmap('hsv', SIM.numRobots)

    plot_handles: Dict[str, Dict[str, Any]] = {}
    robotIds = list(robots.keys())
    
    for i, rid in enumerate(robotIds):
        robot = robots[rid]
        col = colors(i)
        
        plot_handles[rid] = {}
        
        # Goal marker (X)
        ax.plot(robot.goal[0], robot.goal[1], marker='x', color=col, markersize=15, linewidth=2)
        # Start marker (*)
        ax.plot(robot.pose[0], robot.pose[1], marker='*', color=col, markersize=10, linewidth=1.5)
        
        # Path line (dashed, transparent)
        plot_handles[rid]['path'], = ax.plot(robot.path.x, robot.path.y, '--', color=col, linewidth=1.0, alpha=0.3)
        # Trajectory line (dotted)
        plot_handles[rid]['traj'], = ax.plot([], [], ':', color=col, linewidth=2.0)
        # Robot position (circle)
        plot_handles[rid]['robot'], = ax.plot([], [], 'o', color=col, markersize=12, markerfacecolor=col)
        # Heading indicator (line)
        plot_handles[rid]['heading'], = ax.plot([], [], '-', color=col, linewidth=2.5)
        
        # NEW: Robot Bounding Box (rectangle)
        plot_handles[rid]['boundingBox'], = ax.plot([], [], '-', color=col, linewidth=1.5, alpha=0.7)
        
        # Safety radius circle
        plot_handles[rid]['safetyCircle'], = ax.plot([], [], ':', color=col, linewidth=1, alpha=0.2)
        
    title_handle = ax.set_title(f'Multi-Robot Simulation | t=0.0s | Robots: 0/{SIM.numRobots} reached goal')

    # --- SIMULATION LOOP ---
    print(f'\nStarting {"REAL-TIME" if SIM.realTimeMode else "FAST"} simulation...\n')
    
    simTime = 0.0
    frameCount = 0
    
    while simTime < SIM.maxTime:
        frame_start_time = time.time()
        
        # 1. Update all Kalman filters with current measurements (No Change)
        for rid, robot in robots.items():
            if robot.reached:
                continue
            
            # Kalman prediction
            F = np.array([
                [1, 0, SIM.dt, 0],
                [0, 1, 0, SIM.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            robot.kf_state = F @ robot.kf_state
            robot.kf_P = F @ robot.kf_P @ F.T + robot.kf_Q
            
            # Kalman update
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            z = robot.pose[:2]
            y_inn = z - H @ robot.kf_state
            S = H @ robot.kf_P @ H.T + robot.kf_R
            K = robot.kf_P @ H.T @ np.linalg.inv(S)
            robot.kf_state = robot.kf_state + K @ y_inn
            robot.kf_P = (np.eye(4) - K @ H) @ robot.kf_P
            
            # Update velocity estimate
            robot.velocity = robot.kf_state[2:]

        # 2. Control and update each robot (No Change)
        numReached = 0
        for rid, robot in robots.items():
            if robot.reached:
                numReached += 1
                continue
            
            # Check if goal reached
            distToGoal = np.sqrt((robot.pose[0] - robot.goal[0])**2 + (robot.pose[1] - robot.goal[1])**2)
            if distToGoal < CFG.posTolerance:
                robot.reached = True
                robot.reachedTime = simTime
                numReached += 1
                print(f'  [{simTime:.1f}s] {rid} reached goal (replans: {robot.replanCount})')
                continue

            # Path replanning
            if (simTime - robot.lastReplanTime) >= CFG.replanInterval:
                other_robots_list = create_other_robots_list(robots, rid)
                obstacles = create_dynamic_obstacles_sim(other_robots_list, CFG)
                
                newPath = rrt_star_planner(robot.pose, robot.goal, obstacles, CFG, SIM.mapSize)
                
                if newPath is not None:
                    robot.path = smooth_path_sim(newPath)
                    robot.replanCount += 1
                    
                    # Update path visualization
                    plot_handles[rid]['path'].set_data(robot.path.x, robot.path.y)
                
                robot.lastReplanTime = simTime

            # Pure Pursuit Control
            lookaheadX, lookaheadY, _, crossTrackError = \
                find_lookahead_sim(robot.pose[0], robot.pose[1], robot.path, CFG.lookaheadDist)
            
            # Update error history
            robot.errorHistory.append(crossTrackError)
            if len(robot.errorHistory) > 30:
                robot.errorHistory.pop(0)

            # Recovery mode (simplified logic)
            linVel, angVel = 0.0, 0.0
            if robot.recoveryTimer is not None:
                recoveryElapsed = simTime - robot.recoveryTimer
                if recoveryElapsed < CFG.recoveryDuration:
                    linVel = 0.0
                    angVel = np.deg2rad(30) # Spin slowly
                else:
                    robot.recoveryTimer = None
            
            if robot.recoveryTimer is None:
                # Normal Pure Pursuit control
                dx = lookaheadX - robot.pose[0]
                dy = lookaheadY - robot.pose[1]
                desiredHeading = np.arctan2(dy, dx)
                headingError = wrap_to_pi(desiredHeading - robot.pose[2])
                distToLookahead = np.sqrt(dx**2 + dy**2)
                
                # Formula for angular velocity (based on curvature)
                if distToLookahead > 1e-6:
                    angVel = (2 * CFG.targetLinearVel * np.sin(headingError)) / distToLookahead
                else:
                    angVel = 0.0

                if np.abs(headingError) < np.deg2rad(90):
                    linVel = CFG.targetLinearVel * np.cos(headingError)
                else:
                    linVel = CFG.targetLinearVel * 0.3 # Slow forward

            # Apply velocity limits
            linVel = max(0.0, min(linVel, CFG.maxLinVel))
            angVel = max(-CFG.maxAngVel, min(angVel, CFG.maxAngVel))
            
            # Update robot state (simple kinematic model)
            robot.pose[0] += linVel * np.cos(robot.pose[2]) * SIM.dt
            robot.pose[1] += linVel * np.sin(robot.pose[2]) * SIM.dt
            robot.pose[2] = wrap_to_pi(robot.pose[2] + angVel * SIM.dt)
            
            # Update trajectory
            robot.trajectory.append(robot.pose[:2].copy())
            
        # 3. Update visualization
        if SIM.realTimeMode or frameCount % 5 == 0:
            for rid, robot in robots.items():
                
                # Hide visuals if reached
                if robot.reached:
                    plot_handles[rid]['robot'].set_data([], [])
                    plot_handles[rid]['heading'].set_data([], [])
                    plot_handles[rid]['safetyCircle'].set_data([], [])
                    plot_handles[rid]['boundingBox'].set_data([], [])
                    continue
                
                # Update robot position (circle)
                plot_handles[rid]['robot'].set_data([robot.pose[0]], [robot.pose[1]])
                
                # Update heading indicator
                headingLen = 0.25
                hx = [robot.pose[0], robot.pose[0] + headingLen * np.cos(robot.pose[2])]
                hy = [robot.pose[1], robot.pose[1] + headingLen * np.sin(robot.pose[2])]
                plot_handles[rid]['heading'].set_data(hx, hy)
                
                # Update trajectory
                traj_array = np.array(robot.trajectory)
                plot_handles[rid]['traj'].set_data(traj_array[:, 0], traj_array[:, 1])
                
                # NEW: Update Bounding Box
                boxX, boxY = get_bounding_box_corners(
                    robot.pose[0], 
                    robot.pose[1], 
                    robot.pose[2], 
                    CFG.robotLength, # 30 cm side (parallel to heading)
                    CFG.robotWidth   # 40 cm side (transverse)
                )
                plot_handles[rid]['boundingBox'].set_data(boxX, boxY)
                
                # Update safety circle using the new clearance radius
                theta_circle = np.linspace(0, 2 * np.pi, 30)
                circleX = robot.pose[0] + CFG.robotClearanceRadius * np.cos(theta_circle)
                circleY = robot.pose[1] + CFG.robotClearanceRadius * np.sin(theta_circle)
                plot_handles[rid]['safetyCircle'].set_data(circleX, circleY)

            title_handle.set_text(f'Multi-Robot Simulation | t={simTime:.1f}s | Robots: {numReached}/{SIM.numRobots} reached goal')
            fig.canvas.draw()
            fig.canvas.flush_events()

        # 4. Check termination condition (No Change)
        if numReached == SIM.numRobots:
            print('\n✓ All robots reached their goals!')
            break
            
        # 5. Advance time and maintain real-time rate (No Change)
        simTime += SIM.dt
        frameCount += 1
        
        if SIM.realTimeMode:
            elapsed = time.time() - frame_start_time
            if elapsed < SIM.dt:
                time.sleep(SIM.dt - elapsed)
            
    # --- SIMULATION RESULTS (No Change) ---
    print('\n=== Simulation Complete ===')
    print(f'Total time: {simTime:.1f} seconds')
    
    for rid, robot in robots.items():
        if robot.reached:
            print(f'  {rid}: Reached in {robot.reachedTime:.1f}s ({robot.replanCount} replans)')
        else:
            print(f'  {rid}: Did not reach goal')
            
    print('\nSimulation finished.')
    
    # Keep the plot window open after the simulation finishes
    if SIM.realTimeMode:
        plt.ioff()
    plt.show()

if __name__ == '__main__':
    multi_robot_simulator()