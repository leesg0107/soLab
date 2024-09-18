import numpy as np

class BayesianFilter:
    def __init__(self, prior):
        self.prior = prior  # 초기 belief (prior distribution)

    def update(self, observation, likelihood):
        # Bayesian 업데이트
        # posterior ∝ prior * likelihood
        posterior = self.prior * likelihood[observation]
        posterior /= np.sum(posterior)  # 정규화
        self.prior = posterior  # 업데이트된 belief를 새로운 prior로 설정
        return self.prior

def main():
    # 초기 belief (prior distribution)
    prior = np.array([0.5, 0.5])  # 두 개의 상태 (0, 1)에 대한 균등 분포

    # likelihood 배열 (관찰에 대한 확률)
    likelihood = np.array([[0.9, 0.1],  # 상태 0일 때 관찰 확률
                           [0.1, 0.9]])  # 상태 1일 때 관찰 확률

    # BayesianFilter 인스턴스 생성
    bayesian_filter = BayesianFilter(prior)

    # 관찰 시뮬레이션
    observations = [0, 1, 0, 0, 1]  # 관찰된 상태 시퀀스

    for obs in observations:
        posterior = bayesian_filter.update(obs, likelihood)
        print(f"Updated Belief State: {posterior}")

if __name__ == "__main__":
    main()