# main.py
import sys
import json
import time
import pygame
from env import Environment, MazeGame
from sim_human import SimHuman
from qlearning import QLearningAgent

def run_training(num_episodes, render=False):
    # Load maze and game
    env = Environment('maze_layout.json')
    game = MazeGame(env)
    human = SimHuman(env)

    # Inject human behaviour into the game step
    game._human_choose_action = lambda: human.choose_action(
        game.player_pos, game.key_positions, game.keys_collected
    )

    agent = QLearningAgent()

    # PyGame setup (only once if rendering)
    if render:
        pygame.init()
        CELL_SIZE = 60
        screen = pygame.display.set_mode(
            (env.cols * CELL_SIZE, env.rows * CELL_SIZE)
        )
        pygame.display.set_caption("Training – Q‑Learning Zombie")
        clock = pygame.time.Clock()
        FPS = 40  # speed adjustable

    print(f"Training for {num_episodes} episodes...")
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
                screen.fill((194, 197, 204))           # background

                # Draw static maze walls
                env.draw(screen, CELL_SIZE)

                # Draw keys (uncollected)
                for i, (kpos, collected) in enumerate(
                    zip(game.key_positions, game.keys_collected)
                ):
                    if not collected:
                        pygame.draw.circle(
                            screen, (255, 255, 0),
                            (kpos[1] * CELL_SIZE + CELL_SIZE // 2,
                             kpos[0] * CELL_SIZE + CELL_SIZE // 2),
                            CELL_SIZE // 4
                        )

                # Draw door (always visible)
                dr, dc = game.door_pos
                pygame.draw.rect(screen, (139, 69, 19),
                                 (dc * CELL_SIZE, dr * CELL_SIZE,
                                  CELL_SIZE, CELL_SIZE))

                # Draw player (green) and zombie (red)
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
                clock.tick(FPS)

            # Agent acts, environment responds, learning happens
            action = agent.choose_action(state)
            next_state, reward, done, _ = game.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            step_count += 1

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{num_episodes} completed. "
                  f"Epsilon: {agent.epsilon:.3f}  Steps: {step_count}")

    # Save the trained agent
    agent.save("qtable_zombie.pkl")
    print("Training complete. Q-table saved to qtable_zombie.pkl.")

    if render:
        pygame.quit()


if __name__ == "__main__":
    # Simple manual argument parsing
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 main.py -create")
        print("  python3 main.py -train <episodes> [--render]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "-create":
        # Placeholder for future DB integration
        print("Creating initial zombie agent file...")
        agent = QLearningAgent()
        agent.save("qtable_zombie_init.pkl")
        print("Generated fresh qtable_zombie_init.pkl.")

    elif cmd == "-train":
        # check for --render flag anywhere in args
        render = "--render" in sys.argv

        # find number of episodes: first argument after -train that is a number
        episodes = None
        for i in range(2, len(sys.argv)):
            try:
                episodes = int(sys.argv[i])
                break
            except ValueError:
                continue

        if episodes is None:
            print("Please specify the number of episodes, e.g.: "
                  "python3 main.py -train 1000 [--render]")
            sys.exit(1)

        run_training(episodes, render=render)

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python3 main.py -create   OR   "
              "python3 main.py -train <episodes> [--render]")