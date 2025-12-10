import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# -------------------------------------------------------
# Simulation parameters
# -------------------------------------------------------
N_AGENTS = 12
WORLD_SIZE = 10.0
DT = 0.05
ROBOT_SPEED = 1.0
AVOID_DIST = 1.2
AVOID_GAIN = 2.5
GOAL_TOL = 0.2
WAYPOINT_TOL = 0.3
GRID_RES = 1.0  # 1-unit grid cells

# -------------------------------------------------------
# A* PATH PLANNING
# -------------------------------------------------------
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(start, goal, grid_min, grid_max):
    """A* on an open grid with no obstacles."""
    
    # Convert world pos to grid cell
    def to_grid(p):
        return (int((p[0] - grid_min) / GRID_RES),
                int((p[1] - grid_min) / GRID_RES))
        
    def to_world(cell):
        return np.array([cell[0] * GRID_RES + grid_min + GRID_RES / 2,
                         cell[1] * GRID_RES + grid_min + GRID_RES / 2])

    start_c = to_grid(start)
    goal_c = to_grid(goal)

    # 8-connected grid
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
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(to_world(current))
                current = came_from[current]
            path.append(to_world(start_c))
            return path[::-1]  # reversed

        # Explore neighbors
        for dx, dy in nbrs:
            nbr = (current[0] + dx, current[1] + dy)

            # Stay in bounds
            if not (0 <= nbr[0] <= (grid_max - grid_min) / GRID_RES and
                    0 <= nbr[1] <= (grid_max - grid_min) / GRID_RES):
                continue

            tentative_g = g_score[current] + np.hypot(dx, dy)

            if nbr not in g_score or tentative_g < g_score[nbr]:
                g_score[nbr] = tentative_g
                f = tentative_g + heuristic(nbr, goal_c)
                heapq.heappush(open_set, (f, nbr))
                came_from[nbr] = current

    return [start]  # fallback


# -------------------------------------------------------
# Agent Class
# -------------------------------------------------------
class Robot:
    def __init__(self, idx):
        self.idx = idx
        self.position = np.random.uniform(-WORLD_SIZE, WORLD_SIZE, size=2)
        self.goal = np.random.uniform(-WORLD_SIZE, WORLD_SIZE, size=2)

        # A* path creation
        self.path = astar(
            self.position, self.goal,
            grid_min=-WORLD_SIZE,
            grid_max=+WORLD_SIZE
        )
        self.path_i = 0

    def get_waypoint(self):
        if self.path_i >= len(self.path):
            return self.goal
        return self.path[self.path_i]

    def advance_waypoint(self):
        if np.linalg.norm(self.get_waypoint() - self.position) < WAYPOINT_TOL:
            self.path_i += 1

    def compute_desired_velocity(self):
        wp = self.get_waypoint()
        vec = wp - self.position
        dist = np.linalg.norm(vec)
        if dist < GOAL_TOL:
            return np.zeros(2)
        return (vec / dist) * ROBOT_SPEED

    def avoid_others(self, agents, v_desired):
        v = np.copy(v_desired)
        for other in agents:
            if other.idx == self.idx:
                continue
            diff = self.position - other.position
            d = np.linalg.norm(diff)
            if d < AVOID_DIST and d > 1e-6:
                repulse = (diff / d) * (AVOID_GAIN * (AVOID_DIST - d))
                v += repulse
        return v

    def update(self, agents):
        self.advance_waypoint()
        v_desired = self.compute_desired_velocity()
        v_safe = self.avoid_others(agents, v_desired)
        self.position += v_safe * DT


# -------------------------------------------------------
# Create agents
# -------------------------------------------------------
agents = [Robot(i) for i in range(N_AGENTS)]

# -------------------------------------------------------
# Visualization Setup
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-WORLD_SIZE, WORLD_SIZE)
ax.set_ylim(-WORLD_SIZE, WORLD_SIZE)
ax.set_aspect("equal")
ax.set_title("Robot + Digital Twins with A* Path Planning")

points, = ax.plot([], [], "bo", markersize=8)
goals, = ax.plot([], [], "rx", markersize=8)
paths_plots = [ax.plot([], [], "g--", linewidth=1)[0] for _ in agents]

def init():
    return [points, goals] + paths_plots

# -------------------------------------------------------
# Animation update
# -------------------------------------------------------
def update(frame):
    for agent in agents:
        agent.update(agents)

    # robot positions
    xs = [a.position[0] for a in agents]
    ys = [a.position[1] for a in agents]

    # goals
    gx = [a.goal[0] for a in agents]
    gy = [a.goal[1] for a in agents]

    points.set_data(xs, ys)
    goals.set_data(gx, gy)

    # plot paths
    for pl, agent in zip(paths_plots, agents):
        pts = np.array(agent.path)
        pl.set_data(pts[:, 0], pts[:, 1])

    return [points, goals] + paths_plots

anim = FuncAnimation(fig, update, init_func=init, frames=2000, interval=30, blit=True)
plt.show()
