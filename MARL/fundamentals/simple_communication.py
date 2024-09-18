import pygame
import numpy as np
import random

# 환경 설정
GRID_SIZE = 15
CELL_SIZE = 40
VISION_RANGE = 3
MAX_STEPS = 1000

# 색상 정의
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
TRANSPARENT_GRAY = (200, 200, 200, 100)  # 반투명 회색

class ForagingEnvironment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.agents = []
        self.apples = []
        self.place_items()
        self.steps = 0
        self.total_rewards = [0, 0]
        self.messages = [""] * len(self.agents)  # Initialize messages for each agent

    def place_items(self):
        self.agents = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for _ in range(2)]
        for _ in range(8):
            while True:
                x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                if (x, y) not in self.agents and (x, y) not in self.apples:
                    self.apples.append((x, y))
                    break

    def communicate(self):
        # 에이전트가 발견한 사과의 위치를 메시지로 공유
        for i in range(len(self.agents)):
            x, y = self.agents[i]
            self.messages[i] = f"Agent {i} at ({x}, {y})"
            if (x, y) in self.apples:
                self.messages[i] += f"; found apple at ({x}, {y})"  # 사과 발견 메시지 추가

        # 시야가 겹치는 에이전트의 메시지 조정
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if self.is_vision_overlap(i, j):
                    self.messages[j] += f"; Agent {i} has overlapping vision with Agent {j}"

    def is_vision_overlap(self, agent_a_idx, agent_b_idx):
        # 두 에이전트의 시야가 겹치는지 확인
        x_a, y_a = self.agents[agent_a_idx]
        x_b, y_b = self.agents[agent_b_idx]
        
        # 시야 범위 계산
        for i in range(-VISION_RANGE, VISION_RANGE + 1):
            for j in range(-VISION_RANGE, VISION_RANGE + 1):
                if (0 <= x_a + i < GRID_SIZE) and (0 <= y_a + j < GRID_SIZE):
                    if (x_a + i, y_a + j) == (x_b, y_b):
                        return True
        return False

    def get_observation(self, agent_idx):
        x, y = self.agents[agent_idx]
        vision = np.zeros((VISION_RANGE*2+1, VISION_RANGE*2+1), dtype=int)
        for i in range(-VISION_RANGE, VISION_RANGE+1):
            for j in range(-VISION_RANGE, VISION_RANGE+1):
                nx, ny = x+i, y+j
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if (nx, ny) in self.apples:
                        vision[i+VISION_RANGE, j+VISION_RANGE] = 1
                    elif (nx, ny) in self.agents:
                        vision[i+VISION_RANGE, j+VISION_RANGE] = 2
        return vision, self.messages[agent_idx]  # Return vision and message

    def move_agent(self, agent_idx):
        x, y = self.agents[agent_idx]
        target_apple = None
        
        # 다른 에이전트의 메시지에서 사과 위치를 찾기
        for msg in self.messages:
            if "found apple" in msg:
                # 메시지에서 사과 위치를 파싱
                parts = msg.split("; ")
                for part in parts:
                    if "found apple" in part:
                        apple_pos = part.split("at ")[1].strip(")")
                        target_apple = tuple(map(int, apple_pos.strip("()").split(", ")))
                        break

        # 에이전트의 시야에서 사과 찾기
        if not target_apple:
            vision = self.get_observation(agent_idx)[0]  # 시야 정보 가져오기
            for i in range(VISION_RANGE * 2 + 1):
                for j in range(VISION_RANGE * 2 + 1):
                    if vision[i, j] == 1:  # 사과 발견
                        target_apple = (x + (i - VISION_RANGE), y + (j - VISION_RANGE))
                        break
                if target_apple:
                    break

        # 사과가 발견되면 그 방향으로 이동
        if target_apple:
            target_x, target_y = target_apple
            if x < target_x: x += 1
            elif x > target_x: x -= 1
            if y < target_y: y += 1
            elif y > target_y: y -= 1
        else:
            # 랜덤으로 움직임 (사과가 없을 경우)
            action = random.randint(0, 3)
            if action == 0 and y > 0: y -= 1
            elif action == 1 and y < GRID_SIZE-1: y += 1
            elif action == 2 and x > 0: x -= 1
            elif action == 3 and x < GRID_SIZE-1: x += 1

        self.agents[agent_idx] = (x, y)

        if (x, y) in self.apples:
            self.apples.remove((x, y))
            self.total_rewards[agent_idx] += 1
            return 1
        return 0

    def step(self):
        self.communicate()  # Agents communicate before taking actions
        for i in range(len(self.agents)):
            self.move_agent(i)
        self.steps += 1
        return self.steps >= MAX_STEPS or len(self.apples) == 0

def draw_environment(screen, env):
    # 전체 화면을 어둡게 처리
    dark_surface = pygame.Surface((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE), pygame.SRCALPHA)
    dark_surface.fill((0, 0, 0, 200))  # 반투명 검은색

    screen.fill(WHITE)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

    for apple in env.apples:
        x, y = apple
        pygame.draw.circle(screen, RED, (x*CELL_SIZE+CELL_SIZE//2, y*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//3)

    # 어두운 배경 적용
    screen.blit(dark_surface, (0, 0))

    # 각 에이전트의 시야 표시
    for i, agent in enumerate(env.agents):
        x, y = agent
        vision = env.get_observation(i)
        for vi in range(-VISION_RANGE, VISION_RANGE+1):
            for vj in range(-VISION_RANGE, VISION_RANGE+1):
                nx, ny = x+vi, y+vj
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    rect = pygame.Rect(nx*CELL_SIZE, ny*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    # 더 투명한 회색으로 시야 표시
                    pygame.draw.rect(screen, (200, 200, 200, 50), rect)  # 변경된 부분
        
        # 에이전트 그리기
        agent_color = BLACK if i == 0 else (0, 0, 255)  # agent1은 검정색, agent2는 파란색
        pygame.draw.circle(screen, agent_color, (x*CELL_SIZE+CELL_SIZE//2, y*CELL_SIZE+CELL_SIZE//2), CELL_SIZE//3)

    # 스텝 수와 보상 표시
    font = pygame.font.Font(None, 36)
    text = font.render(f"Steps: {env.steps}", True, BLACK)
    screen.blit(text, (10, 10))
    text = font.render(f"Rewards: {env.total_rewards}", True, BLACK)
    screen.blit(text, (10, 50))

    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE))
    pygame.display.set_caption("Foraging Environment")

    env = ForagingEnvironment()
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        done = env.step()
        draw_environment(screen, env)
        
        if done:
            print(f"Simulation ended. Total steps: {env.steps}")
            print(f"Total rewards: {env.total_rewards}")
            pygame.time.wait(3000)  # 3초 대기
            running = False

        clock.tick(10)  # 10 FPS로 실행

    pygame.quit()

if __name__ == "__main__":
    main()