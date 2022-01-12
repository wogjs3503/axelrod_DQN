# axelrod_DQN

죄수의 딜레마 (iterated prisoner's dillema) 테마 연구 환경인 axelrod repository 환경에서 Deep Q-learning algorithm을 이용하여, 존재하는 다양한 전략들을 학습하고 그에 맞게 대응하는 코드를 작성하였습니다.

서로 같이 협력(C)을 하면 3점, 같이 배반(D)을 하면 -1점을 받고, 상대방과 다르게 행동하면 5점, 1점을 받습니다. Tit-for-Tat과 붙으면 all C, Grudger와 붙으면 all D 전략을 사용하는 것을 확인하였습니다. 다양한 전략들이 존재하는 토너먼트에서도 우수한 성적을 보였습니다.

사용한 neural network는 state를 input으로 받아 q value를 return 하도록 설정했습니다. 구체적으로 opponent의 previous state 10개를 (ex CDDCCDCDDC) input으로 삼고, output으로 Cooperation, Deception에 해당하는 q value 2개를 return 하는 구조입니다. 

2013년 Atari 논문처럼 선택한 action의 q value만을 update 한 후 model을 학습시켰기 때문에, 선택한 action과 관련된 weight만 우세하게 update 되도록 하였습니다. 또한 Replay memory를 사용하여 이전에 쌓아 두었던 data를 sampling하여 q value를 update 하도록 하였습니다. 구체적으로 deque 자료구조를 사용하여 최근 30개의 data만 사용하도록 설정하였습니다. 하지만 2013년 논문에서 등장한, 학습 과정에서의 stability에 기여한다고 알려져 있는 target neural network는 사용하지 않았습니다. (정확히 이해가 가지 않아서…)

DQN에서 tensorflow의 keras를 사용하였고 코드를 작성하기 전, https://dnddnjs.gitbooks.io/rl/content/cover.html 를 통해 강화학습의 기초를 공부하였습니다. 저자분께 감사인사를 드립니다. 그리고 프로젝트를 소개하고 지도해주신 사수님 주하람 선배님께도 감사인사 드립니다.

