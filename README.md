# axelrod_DQN

죄수의 딜레마 (iterated prisoner's dillema) 테마 연구 환경인 axelrod repository 환경에서 DQN을 이용하여 존재하는 다양한 전략들을 학습하고 그에 맞게 대응하는 코드를 작성하였습니다.

서로 같이 협력(C)을 하면 3점, 같이 배반(D)을 하면 -1점을 받고, 상대방과 다르게 행동하면 5점, 1점을 받습니다. Tit-for-Tat과 붙으면 all C, Grudger와 붙으면 all D 전략을 사용하는 것을 확인하였습니다. 다양한 전략들이 존재하는 토너먼트에서도 우수한 성적을 보였습니다.

DQN에서 tensorflow의 keras를 사용하였고 코드를 작성하기 전, https://dnddnjs.gitbooks.io/rl/content/cover.html 를 통해 강화학습의 기초를 공부하였습니다. 저자분께 감사인사를 드립니다. 그리고 프로젝트를 소개하고 지도해주신 사수님 주하람 선배님께도 감사인사 드립니다.
