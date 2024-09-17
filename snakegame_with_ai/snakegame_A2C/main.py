import torch
from model import Model
from agent import A2CAgent
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
import numpy as np

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or
        (dir_l and game.is_collision(point_l)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)) or
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_l and game.is_collision(point_d)),

        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Food location
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
        ]

    return np.array(state, dtype=int)

def main():
    game = SnakeGameAI()
    model = Model()
    agent = A2CAgent(model)

    # 이전에 저장된 모델이 있다면 로드
    agent.load_model_if_previously_saved()

    num_games = 1000
    batch_size = 32

    for game_num in range(num_games):
        game.reset()
        state = get_state(game)
        done = False
        score = 0

        while not done:
            observations = []
            actions = []
            rewards = []
            dones = []
            values = []

            for _ in range(batch_size):
                action_id, value = agent.model.action_value(state)
                action = agent.model.all_possible_actions_in_game[action_id]

                # Convert action to game input
                if action == Direction.RIGHT:
                    game_action = [0, 1, 0]
                elif action == Direction.LEFT:
                    game_action = [0, 0, 1]
                else:  # UP or DOWN (straight)
                    game_action = [1, 0, 0]

                reward, done, score = game.play_step(action)
                next_state = get_state(game)

                observations.append(state)
                actions.append(action_id)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                state = get_state(game).reshape(-1)

                if done:
                    break

            # Convert to numpy arrays
            observations = np.array(observations)
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)
            values = np.array(values)

            # Calculate returns and advantages
            returns, advantages = agent._returns_advantages(rewards, dones, values, 0)  # 0 as next_value for terminal state

            # Prepare data for training
            acts_and_advs = np.concatenate([actions[:, None], advantages[:, None]], axis=-1)

            # Train the agent
            loss = agent.train(observations, acts_and_advs, returns)

        print(f"Game {game_num + 1}, Score: {score}, Loss: {loss}")

        # 주기적으로 모델 저장
        if (game_num + 1) % 100 == 0:
            agent.save_model()

    # 학습 완료 후 모델 저장
    agent.save_model()

if __name__ == "__main__":
    main()
