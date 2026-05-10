import os
import sys
import json
import pygame
from env import Environment, MazeGame
from sim_human import SimHuman
from qlearning import QLearningAgent
from datetime import datetime

def run_episodes(agent, num_episodes, render=False, fps=10, window_title="Maze"):
    """
    Run one or more episodes using the given agent.
    Returns after all episodes finish. Does NOT save the agent.
    """
    # Set up the environment and simulated human once
    env = Environment('maze_layout.json')
    game = MazeGame(env)
    human = SimHuman(env)

    # Connect human to the game
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
        done = False
        step_count = 0

        while not done:
            # --- Optional rendering ---
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

            agent.update(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
        
        agent.decay_epsilon()

        # Check if zombie caught the human this episode
        if game.zombie_pos == game.player_pos:
            block_wins += 1

        if num_episodes >= 100 and (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{num_episodes} completed. "
                  f"Epsilon: {agent.epsilon:.3f}  Steps: {step_count}   "
                  f"Zombie wins: {block_wins}/100")
            block_wins = 0
            
    if render:
        pygame.quit()

def cmd_create():
    agent = QLearningAgent()
    agent.save("qtable_zombie_init.pkl")
    print("Created fresh qtable_zombie_init.pkl.")

def cmd_train(episodes, render):
    agent = QLearningAgent()
    run_episodes(agent, episodes, render=render, fps=10,
                 window_title="Training – Q‑Learning Zombie")
    # Save (you can later change to include episode count)
    agent.save("qtable_zombie.pkl")
    print(f"Training complete. Saved to qtable_zombie.pkl.")

def cmd_play(filename):
    agent = QLearningAgent()
    agent.load(filename)
    current_count = agent.total_episodes

    # Run exactly ONE episode with rendering
    run_episodes(agent, 1, render=True, fps=10,
                 window_title=f"Episode {current_count + 1}")

    # Increment and save with updated filename
    agent.total_episodes += 1
    new_filename = f"qtable_zombie_{agent.total_episodes}.pkl"
    agent.save(new_filename)
    print(f"Episode finished. Saved to {new_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 main.py -create")
        print("  python3 main.py -train <episodes> [--render]")
        print("  python3 main.py -play <file.pkl>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "-create":
        cmd_create()

    elif cmd == "-train":
        render = "--render" in sys.argv
        episodes = None
        for arg in sys.argv[2:]:
            try:
                episodes = int(arg)
                break
            except ValueError:
                continue
        if episodes is None:
            print("Please specify the number of episodes, e.g.: python3 main.py -train 1000 [--render]")
            sys.exit(1)
        cmd_train(episodes, render)

    elif cmd == "-play":
        if len(sys.argv) != 3:
            print("Please provide a .pkl file, e.g.: python3 main.py -play qtable_zombie_1000.pkl")
            sys.exit(1)
        cmd_play(sys.argv[2])

    else:
        print(f"Unknown command: {cmd}")