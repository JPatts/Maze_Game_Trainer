import json
import sys
import pygame

class Environment:
    def __init__(self, json_path):
        # Load the JSON file into the instance
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.rows = data['rows']
        self.cols = data['cols']
        self.grid = data['grid']                     # 2D list of cells
        self.key_positions = data.get('keyPositions', [])
        self.door_position = data.get('doorPositions', None)
        self.zombie_start = data.get('zombieStart', None)
        self.player_start = data.get('playerStart', None)

    def draw(self, screen, cell_size, wall_color=(0,0,0), wall_width=2):
        """
        Draw the maze (walls only) onto a Pygame surface.
        screen: pygame.Surface
        cell_size: int – size of each cell in pixels
        wall_color: RGB tuple
        wall_width: line thickness
        """
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.grid[row][col]
                walls = cell['walls']   # [top, right, bottom, left]

                x = col * cell_size
                y = row * cell_size

                # Top wall
                if walls[0]:
                    pygame.draw.line(screen, wall_color, (x, y), (x + cell_size, y), wall_width)
                # Right wall
                if walls[1]:
                    pygame.draw.line(screen, wall_color, (x + cell_size, y), (x + cell_size, y + cell_size), wall_width)
                # Bottom wall
                if walls[2]:
                    pygame.draw.line(screen, wall_color, (x, y + cell_size), (x + cell_size, y + cell_size), wall_width)
                # Left wall
                if walls[3]:
                    pygame.draw.line(screen, wall_color, (x, y), (x, y + cell_size), wall_width)

    def draw_special_points(self, screen, cell_size):
        """
        Draw markers for keys, door, player start, zombie start.
        You can customise the colours and shapes.
        """
        # Helper to convert {row, col} to centre of cell
        def cell_center(pos):
            return (pos['col'] * cell_size + cell_size // 2,
                    pos['row'] * cell_size + cell_size // 2)

        # Keys – small yellow circles
        for key in self.key_positions:
            center = cell_center(key)
            pygame.draw.circle(screen, (255, 255, 0), center, cell_size // 4)

        # Door – brown rectangle
        if self.door_position:
            door = self.door_position
            rect = pygame.Rect(door['col'] * cell_size, door['row'] * cell_size,
                               cell_size, cell_size)
            pygame.draw.rect(screen, (139, 69, 19), rect)

        # Player start – green circle
        if self.player_start:
            center = cell_center(self.player_start)
            pygame.draw.circle(screen, (0, 255, 0), center, cell_size // 3)

        # Zombie start – red circle
        if self.zombie_start:
            center = cell_center(self.zombie_start)
            pygame.draw.circle(screen, (255, 0, 0), center, cell_size // 3)

    # wall logic
    def is_valid_move(self, from_row, from_col, to_row, to_col):
        """Return True if moving from (from_row,from_col) to (to_row,to_col) is legal."""
        if not (0 <= to_row < self.rows and 0 <= to_col < self.cols):
            return False
        walls = self.grid[from_row][from_col]['walls']
        # moving up   -> check top wall (index 0)
        if to_row == from_row - 1 and to_col == from_col:
            return not walls[0]
        # moving down -> check bottom wall (index 2)
        if to_row == from_row + 1 and to_col == from_col:
            return not walls[2]
        # moving left -> check left wall (index 3)
        if to_row == from_row and to_col == from_col - 1:
            return not walls[3]
        # moving right-> check right wall (index 1)
        if to_row == from_row and to_col == from_col + 1:
            return not walls[1]
        return False
    
class MazeGame:
    def __init__(self, env):
        """
        env: an Enviornment instance that holds grid layout 
        """
        self.env = env
        self.rows = env.rows
        self.cols = env.cols

        # starting positions
        self.zombie_start = (env.zombie_start['row'], env.zombie_start['col'])
        self.player_start = (env.player_start['row'], env.player_start['col'])

        # key & door positions
        self.key_positions = [ (k['row'], k['col']) for k in env.key_positions ]
        self.door_pos = (env.door_position['row'], env.door_position['col'])

        self.reset()

    def reset(self):
        self.zombie_pos = self.zombie_start
        self.player_pos = self.player_start

        self.keys_collected = [False] * len(self.key_positions)

        self.door_open = False
        self.done = False

        return self._get_state()

    def step(self, zombie_action):
        """
        Take one environment step
        zombie action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        Returns: next_state, reward, done, info_dict
        """
        # zombie moves
        new_zombie_pos = self._move(self.zombie_pos, zombie_action)
        self.zombie_pos = new_zombie_pos

        # human moves
        human_action = self._human_choose_action()
        new_player_pos = self._move(self.player_pos, human_action)
        self.player_pos = new_player_pos

        # check key collect
        self._check_key_collection()

        # check win condition: player has all keys and at door
        if self.player_pos == self.door_pos and all(self.keys_collected):
            self.done = True
            reward = -10.0
            return self._get_state(), reward, self.done, {"win": True}
        
        # check loss condition
        if self.zombie_pos == self.player_pos:
            self.done
            reward = 20
            return self._get_state(), reward, self.done, {} 
        
        # small cost of reward per step to encourage getting to human quicker
        reward = -0.1


        return self._get_state(), reward, self.done, {}
    
    def _get_state(self):
        """ Returns a compact state representation """
        return {
            'zombie_row': self.zombie_pos[0],
            'zombie_col': self.zombie_pos[1],
            'player_row': self.player_pos[0],
            'player_col': self.player_pos[1],
            'keys_collected': sum(self.keys_collected),
        }
    
    def _move(self, pos, action):
        # Apply action to a position, with repsect to walls
        row, col = pos
        moves = {0: (row-1, col), 1: (row,col+1), 2: (row+1, col), 3: (row, col-1)}
        new_row, new_col = moves.get(action, (row,col))
        if self.env.is_valid_move(row, col, new_row, new_col):
            return (new_row, new_col)
        return pos
    
    def _check_key_collection(self):
        # If palyer is on an uncollected key; collect it
        for i, key_pos in enumerate(self.key_positions):
            if not self.keys_collected[i] and self.player_pos == key_pos:
                self.keys_collected[i] = True
    
    def _human_choose_action(self):
        # For a now simple random valid move 
        import random
        actions = list(range(4))
        random.shuffle(actions)
        for a in actions:
            new_pos = self._move(self.player_pos, a)
            if new_pos != self.player_pos:
                return a
        return random.choice(actions)

def main():
    pygame.init()

    # Settings
    CELL_SIZE = 60
    WINDOW_WIDTH = 15 * CELL_SIZE   # cols
    WINDOW_HEIGHT = 15 * CELL_SIZE  # rows

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Maze Viewer")

    # Load the maze
    maze = Environment('maze_layout.json')

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Press ESC to quit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Drawing
        screen.fill((194, 197, 204))          # white background
        maze.draw(screen, CELL_SIZE)          # draw walls
        maze.draw_special_points(screen, CELL_SIZE)  # optional markers

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()