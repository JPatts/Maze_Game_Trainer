# sim_human.py
import heapq
import random

class SimHuman:
    def __init__(self, env):
        self.env = env          # Environment instance (for is_valid_move)
        self.rows = env.rows
        self.cols = env.cols
        self.error_prob = 0.15  # 15% chance of sub-optimal move
        self.hesitation_prob = 0.1

    def choose_action(self, current_pos, key_positions, keys_collected):
        """
        current_pos: (row, col) of human
        key_positions: list of (row, col) for all keys
        keys_collected: list of bools indicating which keys are already collected
        Returns: action 0-3 (UP, RIGHT, DOWN, LEFT)
        """
        # Find next uncollected key (random choice among uncollected)
        uncollected = [key_positions[i] for i, collected in enumerate(keys_collected) if not collected]
        if not uncollected:
            # All keys collected → head for door
            target = self.env.door_position
            target = (target['row'], target['col'])
        else:
            target = random.choice(uncollected)

        # Compute shortest path using A*
        path = self._astar(current_pos, target)
        if len(path) < 2:
            # Already at target or no path → no movement
            return None

        next_cell = path[1]  # first step in path
        action = self._direction_to_action(current_pos, next_cell)

        # --- Simulate human behaviour ---
        # Hesitate: stay still with some probability
        if random.random() < self.hesitation_prob:
            return None   # no movement this turn

        # Error: pick a random valid move instead of the optimal one
        if random.random() < self.error_prob:
            valid_actions = self._get_valid_actions(current_pos)
            if valid_actions:
                return random.choice(valid_actions)
        return action

    def _astar(self, start, goal):
        """Return list of cells [(r,c), ...] from start to goal (inclusive)."""
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break

            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current

        # Reconstruct path
        if goal not in came_from:
            return []   # no path
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    def _get_neighbors(self, pos):
        row, col = pos
        candidates = [(row-1,col), (row,col+1), (row+1,col), (row,col-1)]
        return [c for c in candidates if self.env.is_valid_move(row, col, c[0], c[1])]

    def _get_valid_actions(self, pos):
        row, col = pos
        actions = []
        for i, (dr, dc) in enumerate([(-1,0), (0,1), (1,0), (0,-1)]):
            nr, nc = row+dr, col+dc
            if self.env.is_valid_move(row, col, nr, nc):
                actions.append(i)
        return actions

    def _direction_to_action(self, from_cell, to_cell):
        dr = to_cell[0] - from_cell[0]
        dc = to_cell[1] - from_cell[1]
        if dr == -1: return 0
        if dr == 1: return 2
        if dc == -1: return 3
        if dc == 1: return 1
        return None