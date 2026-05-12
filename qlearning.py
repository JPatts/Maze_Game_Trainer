import json
import os
import random
import pickle

class QLearningAgent:
    def __init__(self, action_space=4, alpha=0.1, gamma=0.9, epsilon_start=1.0,epsilon_end=0.05, epsilon_decay=0.9995):
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay = epsilon_decay 
        self.total_episodes = 0

    def _state_to_key(self, state):
        """
        Convert state dictionary to a hasable key for q-table lookup
        Includes direction to human, distance bucket, wall info, and keys collected
        """

        if state is None:
            return None
        
        zombie_row = state['zombie_row']
        zombie_col = state['zombie_col']

        # wall detection - which direction are blocked
        walls = (
            not state['env'].is_valid_move(zombie_row, zombie_col, zombie_row - 1, zombie_col),
            not state['env'].is_valid_move(zombie_row, zombie_col, zombie_row, zombie_col + 1),
            not state['env'].is_valid_move(zombie_row, zombie_col, zombie_row + 1, zombie_col),
            not state['env'].is_valid_move(zombie_row, zombie_col, zombie_row, zombie_col - 1),
        )

        walls_str = ",".join(str(int(w)) for w in walls)
        return f"{zombie_row}|{zombie_col}|{walls_str}|{state['keys_collected']}"

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0] * self.action_space

        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            q_values = self.q_table[key]
            max_q = max(q_values)
            # break ties randomly
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Q-learning update."""
        key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = [0.0] * self.action_space
        if next_key not in self.q_table:
            self.q_table[next_key] = [0.0] * self.action_space

        current_q = self.q_table[key][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_key])
        self.q_table[key][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        if not filename.endswith(".json"):
            filename = os.path.splitext(filename)[0] + ".json"
        with open(filename,'w') as f:
            json.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'total_episodes': self.total_episodes,
            }, f, indent=2)

    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.q_table = data.get('q_table', {})
        self.epsilon = data.get('epsilon', 1.0)
        self.total_episodes = data.get('total_episodes', 0)