import numpy as np

class BeliefState:
    def __init__(self, num_states):
        self.belief = np.ones(num_states) / num_states  # 초기 belief state는 균등 분포

    def update(self, observation, likelihood):
        # Bayesian 업데이트
        self.belief = self.belief * likelihood[observation]
        self.belief /= np.sum(self.belief)  # 정규화

    def get_belief(self):
        return self.belief

class Environment:
    def __init__(self, num_states):
        self.num_states = num_states
        self.true_state = np.random.randint(num_states)  # 실제 상태

    def get_observation(self):
        # 현재 상태를 반환
        return self.true_state

    def step(self):
        # 상태를 랜덤으로 변경
        self.true_state = np.random.randint(self.num_states)

class Agent:
    def __init__(self, num_states):
        self.belief_state = BeliefState(num_states)

    def act(self, environment):
        observation = environment.get_observation()
        # 상태에 따라 likelihood를 다르게 설정
        likelihood = np.array([0.9, 0.1] if observation == 0 else [0.1, 0.9])
        self.belief_state.update(observation, likelihood)
        return self.belief_state.get_belief()

def main():
    num_states = 2  # 예시: 두 개의 상태 (0, 1)
    env = Environment(num_states)
    agent = Agent(num_states)

    for _ in range(10):  # 10번의 시간 단계
        belief = agent.act(env)
        print(f"Updated Belief State: {belief}")
        env.step()  # 환경의 상태를 변경

if __name__ == "__main__":
    main()