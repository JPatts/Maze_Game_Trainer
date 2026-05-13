import random
from collections import deque

class SimHuman:
    def __init__(self, env, error_prob=0.25, hesitation_prob=0.50):
        self.env = env
        self.rows = env.rows
        self.cols = env.cols
        self.error_prob = error_prob
        self.hesitation_prob = hesitation_prob
        self.error_interval = 3       # check for error after this many moves
        self.hesitation_interval = 3    # check for hesitation after this many moves

        # Internal state
        self.current_path = []          # list of (r,c) from current position to target
        self.current_target = None      # (r,c) of key/door/flee point
        self.fleeing = False
        self.steps_since_error = 0
        self.steps_since_hesitation = 0

    def reset(self):
        self.current_path = []
        self.current_target = None
        self.fleeing = False
        self.steps_since_error = 0
        self.steps_since_hesitation = 0

    def choose_action(self, current_pos, key_positions, keys_collected, zombie_pos):
        """
        Decide the next move for the simulated human.

        First plans a path to the current objective (key or door). If the zombie
        is too close, overrides with a flee maneuver. Otherwise follows the
        planned path while applying human‑like hesitation and random errors.
        Never steps onto the zombie cell.

        Returns:
            int or None: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, or None to stay in place.
        """
        self.zombie_pos = zombie_pos

        self._update_objective(current_pos, key_positions, keys_collected, zombie_pos)

        if self._flee_if_too_close(current_pos, zombie_pos):
            if self.current_path and len(self.current_path) >= 2:
                next_cell = self.current_path[1]
                if next_cell == zombie_pos:
                    action = self._safe_retreat_action(current_pos, zombie_pos)
                    if action is not None:
                        self.current_path.pop(0)
                    return action
                action = self._direction_to_action(current_pos, next_cell)
                if action is not None and action == self._direction_to_action(current_pos, next_cell):
                    self.current_path.pop(0)
                return action
            else:
                pass

        action = None
        if self.current_path and len(self.current_path) >= 2:
            next_cell = self.current_path[1]
            action = self._direction_to_action(current_pos, next_cell)

        if self.current_path and len(self.current_path) >= 2 and next_cell == zombie_pos:
            safe_actions = []
            current_r, current_c = current_pos
            for act, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                nr, nc = current_r + dr, current_c + dc
                if self.env.is_valid_move(current_r, current_c, nr, nc):
                    if (nr, nc) != zombie_pos:
                        new_dist = abs(nr - zombie_pos[0]) + abs(nc - zombie_pos[1])
                        old_dist = abs(current_r - zombie_pos[0]) + abs(current_c - zombie_pos[1])
                        if new_dist > old_dist:
                            safe_actions.append(act)
            if safe_actions:
                action = random.choice(safe_actions)
            else:
                action = None
            self.current_path = []

        if action is not None:
            self.steps_since_error += 1
            self.steps_since_hesitation += 1

            if self.steps_since_error >= self.error_interval:
                if random.random() < self.error_prob:
                    valid_actions = self._get_valid_actions(current_pos)
                    if valid_actions:
                        action = random.choice(valid_actions)
                        self.current_path = []
                self.steps_since_error = 0

            if self.steps_since_hesitation >= self.hesitation_interval:
                if random.random() < self.hesitation_prob:
                    action = None
                self.steps_since_hesitation = 0

        if (action is not None and self.current_path
                and action == self._direction_to_action(current_pos, self.current_path[1])):
            self.current_path.pop(0)

        return action
    
    def _find_retreat_target(self,start,zombie_pos):
        from collections import deque
        visited = {start}
        q = deque([start])
        best_dist = -1
        best_cell = start
        while q:
            cell = q.popleft()
            d = abs(cell[0] - zombie_pos[0]) + abs(cell[1] - zombie_pos[1])
            if d > best_dist:
                best_dist = d
                best_cell = cell
            for nb in self._get_neighbors(cell):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return best_cell if best_dist > 0 else None
    
    def _update_objective(self, current_pos, key_positions, keys_collected, zombie_pos):
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
        self.current_path = self._astar(current_pos, target, zombie_pos)

    def _astar(self, start, goal, zombie_pos=None):
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
                move_cost = 1 + self._zombie_penalty(nb, zombie_pos)
                tg = g_score[current] + move_cost
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
    
    def _zombie_penalty(self, cell, zombie_pos):
        if zombie_pos is None:
            return 0
        
        dist = abs(cell[0] - zombie_pos[0]) + abs(cell[1] - zombie_pos[1])
        if dist <= 1:
            return 50 
        elif dist == 2:
            return 10
        elif dist == 3:
            return 5
        return 0
    
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
    
    def _flee_if_too_close(self, current_pos, zombie_pos):
        """If zombie is within 2 cells, pick a far-away cell (BFS) and go there."""
        dist = abs(current_pos[0] - zombie_pos[0]) + abs(current_pos[1] - zombie_pos[1])
        if dist > 2:
            return False   # not too close

        # Already fleeing?  If we've reached the target (or it's no longer valid),
        # clear the fleeing flag and let normal logic resume.
        if self.fleeing and self.current_target is not None:
            if current_pos == self.current_target:
                self.fleeing = False
                self.current_path = []
                self.current_target = None
            return True   # keep fleeing

        # Start a new flee
        self.fleeing = True
        # Find the cell that is farthest from the zombie (simple BFS)
        retreat = self._find_retreat_target(current_pos, zombie_pos)
        if retreat and retreat != current_pos:
            self.current_target = retreat
            self.current_path = self._astar(current_pos, retreat, zombie_pos)
            return True

        # If no safe cell found, just continue normally
        self.fleeing = False
        return False
    
    def _safe_retreat_action(self, current_pos, zombie_pos):
        """
        Return an action that moves the human away from the zombie.
        Prefers moves that increase Manhattan distance. If none are possible,
        picks any valid move (other than stepping on the zombie).
        Returns None if no safe move exists.
        """
        valid_actions = []
        retreat_actions = []
        current_r, current_c = current_pos
        best_dist = abs(current_r - zombie_pos[0]) + abs(current_c - zombie_pos[1])

        for act, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            nr, nc = current_r + dr, current_c + dc
            if self.env.is_valid_move(current_r, current_c, nr, nc):
                if (nr, nc) == zombie_pos:
                    continue
                dist = abs(nr - zombie_pos[0]) + abs(nc - zombie_pos[1])
                if dist > best_dist:
                    retreat_actions.append(act)
                valid_actions.append(act)

        if retreat_actions:
            return random.choice(retreat_actions)
        if valid_actions:
            return random.choice(valid_actions)
        return None