import random
import pickle

class QLearningAgent:
    def __init__(self, action_space=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _state_to_key(self, state):
        """Convert state dict to a hashable tuple for Q-table lookup."""
        return (state['zombie_row'], state['zombie_col'],
                state['player_row'], state['player_col'],
                state['keys_collected'])

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

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)