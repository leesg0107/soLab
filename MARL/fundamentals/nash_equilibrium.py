import numpy as np

class Game:
    def __init__(self):
        # 보상 행렬 (2x2 게임 예시)
        self.payoff_matrix = np.array([[[3, 3], [0, 5]],  # 에이전트 1의 보상
                                        [[5, 0], [1, 1]]])  # 에이전트 2의 보상

    def get_payoff(self, action1, action2):
        return self.payoff_matrix[0][action1][action2], self.payoff_matrix[1][action1][action2]

class Agent:
    def __init__(self, id):
        self.id = id
        self.policy = np.array([0.5, 0.5])  # 초기 정책 (균등 분포)
    
    def update_policy(self, opponent_action, game, epsilon=0):
        # 현재 정책에 따라 행동 선택
        action = np.random.choice([0, 1], p=self.policy)
        best_response = self.best_response(opponent_action, game)
        
        # ε-Nash equilibrium을 위한 정책 업데이트
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])  # 무작위 행동 선택
        else:
            action = best_response
        
        # 정책 업데이트
        self.policy[action] += 0.1  # 선택한 행동의 확률 증가
        self.policy /= np.sum(self.policy)  # 정규화

    def best_response(self, opponent_action, game):
        # 상대방의 행동에 대한 최적 반응 계산
        payoffs = [game.get_payoff(action, opponent_action)[self.id] for action in range(2)]
        return np.argmax(payoffs)

def main():
    game = Game()
    agent1 = Agent(0)
    agent2 = Agent(1)

    epsilon = 0.3  # ε-Nash equilibrium을 위한 ε 값

    for _ in range(100):  # 100번의 반복
        action1 = np.random.choice([0, 1], p=agent1.policy)
        action2 = np.random.choice([0, 1], p=agent2.policy)

        # 각 에이전트의 정책 업데이트
        agent1.update_policy(action2, game, epsilon)
        agent2.update_policy(action1, game, epsilon)

        print(f"Agent 1 Policy: {agent1.policy}, Agent 2 Policy: {agent2.policy}")

if __name__ == "__main__":
    main()