import os
import sys
import json
import pygame
from env import Environment, MazeGame
from sim_human import SimHuman
from qlearning import QLearningAgent
from datetime import datetime

def new_session_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"qtable_session_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    return folder

def run_episodes(agent, num_episodes, render=False, fps=10, window_title="Maze"):
    """
    Run one or more episodes using the given agent.
    Returns after all episodes finish. Does NOT save the agent.
    """
    env = Environment('maze_layout.json')
    game = MazeGame(env)
    human = SimHuman(env)

    game._human_choose_action = lambda: human.choose_action(
        game.player_pos, game.key_positions, game.keys_collected, game.zombie_pos
    )

    # Pygame initialisation (only if rendering)
    if render:
        pygame.init()
        CELL_SIZE = 60
        screen = pygame.display.set_mode(
            (env.cols * CELL_SIZE, env.rows * CELL_SIZE)
        )
        pygame.display.set_caption(window_title)
        clock = pygame.time.Clock()

    print(f"Running {num_episodes} episode(s)...")

    block_wins = 0 

    for ep in range(num_episodes):
        state = game.reset()
        human.reset()
        done = False
        step_count = 0

        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                screen.fill((194, 197, 204))
                env.draw(screen, CELL_SIZE)

                # Keys (uncollected)
                for i, (kpos, collected) in enumerate(
                    zip(game.key_positions, game.keys_collected)
                ):
                    if not collected:
                        pygame.draw.circle(
                            screen, (255, 255, 0),
                            (kpos[1] * CELL_SIZE + CELL_SIZE // 2,
                             kpos[0] * CELL_SIZE + CELL_SIZE // 2),
                            CELL_SIZE // 4)

                # Door
                dr, dc = game.door_pos
                pygame.draw.rect(screen, (139, 69, 19),
                                 (dc * CELL_SIZE, dr * CELL_SIZE,
                                  CELL_SIZE, CELL_SIZE))

                # Player (green) and zombie (red)
                px, py = game.player_pos
                zx, zy = game.zombie_pos
                pygame.draw.circle(screen, (0, 255, 0),
                                   (py * CELL_SIZE + CELL_SIZE // 2,
                                    px * CELL_SIZE + CELL_SIZE // 2),
                                   CELL_SIZE // 3)
                pygame.draw.circle(screen, (255, 0, 0),
                                   (zy * CELL_SIZE + CELL_SIZE // 2,
                                    zx * CELL_SIZE + CELL_SIZE // 2),
                                   CELL_SIZE // 3)

                pygame.display.flip()
                clock.tick(fps)

            # Agent step and learning
            action = agent.choose_action(state)
            next_state, reward, done, _ = game.step(action)

            prev_dist = abs(state['zombie_row'] - state['player_row']) + \
                        abs(state['zombie_col'] - state['player_col'])
            new_dist = abs(next_state['zombie_row'] - next_state['player_row']) + \
                        abs(next_state['zombie_col'] - next_state['player_col'])
            shaped_reward = reward
            shaped_reward -= 0.01
            shaped_reward += 0.1 * (prev_dist - new_dist)

            agent.update(state, action, shaped_reward, next_state, done)
            state = next_state
            step_count += 1
        
        agent.decay_epsilon()
        agent.decay_alpha()

        # Check if zombie caught the human this episode
        if game.zombie_pos == game.player_pos:
            block_wins += 1

        if num_episodes >= 100 and (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{num_episodes} completed. "
                  f"Epsilon: {agent.epsilon:.3f}  Steps: {step_count}   "
                  f"Zombie wins: {block_wins}/100")
            block_wins = 0

    agent.total_episodes += num_episodes
            
    if render:
        pygame.quit()

def cmd_create():
    agent = QLearningAgent()
    folder = new_session_folder()
    path = os.path.join(folder, "qtable_zombie_init.json")
    agent.save(path)
    print(f"Created fresh {path}")

def cmd_train(episodes, render, start_file):
    """
    If start file is given, load that agent; otherwise start fresh
    After training, save the agent into a NEW timestamped folder.
    Using the cumulative episode count in the filename
    """
    agent = QLearningAgent()
    if start_file:
        if not os.path.exists(start_file):
            print(f"Error: start file `{start_file}` not found.")
            sys.exit(1)
        agent.load(start_file)
        print(f"Loaded agent from {start_file}. "
              f"Already trained: {agent.total_episodes} episodes")
    else:
        print("Starting training from scratch (epsilon = 1.0)")
    
    run_episodes(agent, episodes, render=render, fps=10, window_title="Training Qlearning Zombie")

    # create new session folder and save
    folder = new_session_folder()
    new_filename = f"qtable_zombie_{agent.total_episodes}.json"
    save_path = os.path.join(folder,new_filename)
    agent.save(save_path)
    print(f"Training complete. Saved to {save_path}")

def cmd_play(filename):
    if not os.path.exists(filename):
        print("Error: file '{filename}' not found.")
        sys.exit(1)

    agent = QLearningAgent()
    agent.load(filename)
    current_count = agent.total_episodes

    result = run_episodes(agent, 1, render=True, fps=10, window_title=f"Episode {current_count + 1}")
    print(result)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 main.py -create")
        print("  python3 main.py -train <episodes> [--render]")
        print("  python3 main.py -train <episodes> <path/to/start.json> [--render]")
        print("  python3 main.py -play <file.pkl>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "-create":
        cmd_create()

    elif cmd == "-train":
        args = sys.argv[2:]
        render = False
        episodes = None
        start_file = None

        # filter out --render and find episode count and optional .pkl file
        for arg in args:
            if arg == "-- render":
                render = True
            elif arg.endswith(".json"):
                start_file = arg
            else:
                try:
                    episodes = int(arg)
                except ValueError:
                    print(f"Unkown Argument: {arg}")
                    sys.exit(1)
        
        if episodes is None:
            print("Please specify the number of epsidoes, e.g.: Python3 main.py -train 1000")
            sys.exit(1)

        cmd_train(episodes, render, start_file)
    
    elif cmd == "-play":
        if len(sys.argv) != 3:
            print("Please prvide a .pkl file, e.g.: python3 main.py -play qtable_zombie_1000.pkl")
            sys.exit(1)
        cmd_play(sys.argv[2])
    
    else:
        print(f"Unkown command: {cmd}")

if __name__ == "__main__":
    main()