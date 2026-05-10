import random
from collections import deque

class SimHuman:
    def __init__(self, env, error_prob=0.25, hesitation_prob=0.25):
        self.env = env
        self.rows = env.rows
        self.cols = env.cols
        self.error_prob = error_prob
        self.hesitation_prob = hesitation_prob
        self.error_interval = 2       # check for error after this many moves
        self.hesitation_interval = 2    # check for hesitation after this many moves

        # Internal state
        self.current_path = []          # list of (r,c) from current position to target
        self.current_target = None      # (r,c) of key/door/flee point
        self.fleeing = False
        self.steps_since_error = 0
        self.steps_since_hesitation = 0

    def choose_action(self, current_pos, key_positions, keys_collected, zombie_pos):
        """
        Returns an action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, or None (no move).
        """
        self._update_objective(current_pos, key_positions, keys_collected)

        action = None
        if self.current_path and len(self.current_path) >= 2:
            # Move from current_path[0] (us) to current_path[1]
            next_cell = self.current_path[1]
            action = self._direction_to_action(current_pos, next_cell)

        # Apply human‑like mistakes for move interval
        if action is not None:
            self.steps_since_error += 1
            self.steps_since_hesitation += 1

            # Error: random valid move every N moves?
            if self.steps_since_error >= self.error_interval:
                if random.random() < self.error_prob:
                    valid_actions = self._get_valid_actions(current_pos)
                    if valid_actions:
                        action = random.choice(valid_actions)
                        # discard current path because we just left it
                        self.current_path = []
                self.steps_since_error = 0

            # Hesitation: no move every N moves
            if self.steps_since_hesitation >= self.hesitation_interval:
                if random.random() < self.hesitation_prob:
                    action = None
                self.steps_since_hesitation = 0

        # If we actually moved along the planned path, advance the path
        if action is not None and self.current_path and action == self._direction_to_action(
            current_pos, self.current_path[1]
        ):
            # We are following the path – pop the start so next call is aligned
            self.current_path.pop(0)

        return action
    
    def _update_objective(self, current_pos, key_positions, keys_collected):
        """Pick a new target (key or door) if needed, and compute A* path."""
        # If we already have a valid target and path, don't change
        if self.current_target is not None and self.current_path:
            # Check if target is still valid (key still uncollected, door unchanged)
            if self.current_target == self.env.door_position_as_tuple():  # helper needed
                if not all(keys_collected):
                    # door not yet openable – abandon target
                    self.current_path = []
                    self.current_target = None
            else:
                # it's a key – still there?
                for i, kp in enumerate(key_positions):
                    if kp == self.current_target and not keys_collected[i]:
                        return   # still valid
                # key collected or disappeared – abandon
                self.current_path = []
                self.current_target = None

        # Need a new target
        uncollected = [key_positions[i] for i, c in enumerate(keys_collected) if not c]
        if uncollected:
            target = random.choice(uncollected)
        else:
            # All keys collected → head for door
            door = self.env.door_position
            target = (door['row'], door['col'])

        self.current_target = target
        self.current_path = self._astar(current_pos, target)

    def _astar(self, start, goal):
        import heapq
        def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break
            for nb in self._get_neighbors(current):
                tg = g_score[current] + 1
                if nb not in g_score or tg < g_score[nb]:
                    g_score[nb] = tg
                    f = tg + heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))
                    came_from[nb] = current
        if goal not in came_from:
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path
    
    def _get_neighbors(self, pos):
        r, c = pos
        cand = [(r-1,c), (r,c+1), (r+1,c), (r,c-1)]
        return [(nr,nc) for nr,nc in cand if self.env.is_valid_move(r, c, nr, nc)]

    def _get_valid_actions(self, pos):
        actions = []
        r, c = pos
        for i, (dr, dc) in enumerate([(-1,0),(0,1),(1,0),(0,-1)]):
            nr, nc = r+dr, c+dc
            if self.env.is_valid_move(r, c, nr, nc):
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