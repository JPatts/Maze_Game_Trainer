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
        drow = state['player_row'] - state['zombie_row']
        dcol = state['player_col'] - state['zombie_col']
        manhattan = abs(drow) + abs(dcol)
        dist_bin = min(manhattan // 2, 8)
        # Direction quadrant 
        dir_row = (1 if drow > 0 else (-1 if drow < 0 else 0))
        dir_col = (1 if dcol > 0 else (-1 if dcol < 0 else 0))
        return (dir_row, dir_col, dist_bin, state['keys_collected'])

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
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'total_episodes': self.total_episodes
            }, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            # New format: dictionary with possible keys
            # 'q_table', 'epsilon', 'total_episodes'
            self.q_table = data.get('q_table', {})
            self.epsilon = data.get('epsilon', 1.0)
            self.total_episodes = data.get('total_episodes', 0)
            # legacy keys (if they exist in some old saved files)
            if 'qtable' in data:          # handle very old format
                self.q_table = data['qtable']
            if 'episodes_trained' in data:
                self.total_episodes = data['episodes_trained']
        else:
            # Old format: raw Q-table dictionary
            self.q_table = data
            self.epsilon = 1.0
            self.total_episodes = 0