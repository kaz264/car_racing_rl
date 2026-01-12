# 🏎️ CarRacing AI with Reinforcement Learning

OpenAI Gym의 CarRacing 환경에서 PPO 알고리즘을 사용하여 자율주행 AI를 학습시키는 프로젝트입니다.

## 📋 프로젝트 소개

이 프로젝트는 강화학습(Reinforcement Learning)을 사용하여 자동차가 스스로 운전하는 법을 배우도록 합니다.
- **알고리즘**: PPO (Proximal Policy Optimization)
- **환경**: Gymnasium CarRacing-v3
- **입력**: 96x96 RGB 이미지 (게임 화면)
- **출력**: 핸들, 가속, 브레이크 제어

## 🚀 빠른 시작 (Google Colab)

### 1. 학습하기
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaz264/car_racing_rl/blob/master/colab_train.ipynb)

`colab_train.ipynb`를 Colab에서 열어 실행하세요.
- 학습 시간: 약 30분~1시간 (GPU 사용 시)
- 100,000 타임스텝 학습
- 자동으로 모델 저장

### 2. 테스트하기
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaz264/car_racing_rl/blob/master/colab_test.ipynb)

`colab_test.ipynb`를 Colab에서 열어 실행하세요.
- 학습된 AI의 주행 영상 확인
- 3개 에피소드 자동 녹화
- MP4 파일 다운로드 가능

## 💻 로컬 환경에서 실행

### 설치
```bash
# 저장소 클론
git clone https://github.com/kaz264/car_racing_rl.git
cd car_racing_rl

# 의존성 설치
pip install -r requirements.txt
```

### 학습
```bash
python train_car_racing.py
```

### 테스트
```bash
python test_car_racing.py
```

## 📁 프로젝트 구조

```
car_racing_rl/
├── train_car_racing.py      # 로컬 학습 스크립트
├── test_car_racing.py        # 로컬 테스트 스크립트
├── colab_train.ipynb         # Colab 학습 노트북
├── colab_test.ipynb          # Colab 테스트 노트북
├── requirements.txt          # 의존성 패키지
├── README.md                 # 프로젝트 설명
└── (생성되는 폴더들)
    ├── checkpoints/          # 중간 저장 모델
    ├── logs/                 # TensorBoard 로그
    ├── models/               # 최종 학습 모델
    └── video_output/         # 주행 테스트 영상
```

## 🎯 학습 과정

### 1. 경험 수집 (2048 스텝)
AI가 게임을 플레이하며 경험을 수집합니다.
- 화면 이미지 관찰
- 행동 선택 (핸들, 가속, 브레이크)
- 보상 받기

### 2. 학습 (10 에포크)
수집한 경험으로 신경망을 업데이트합니다.
- 좋은 행동의 확률 증가
- 나쁜 행동의 확률 감소

### 3. 반복
100,000 스텝 동안 반복하며 점진적으로 개선됩니다.

## 📊 보상 체계

- **+점수**: 트랙의 새로운 타일 방문
- **-0.1**: 매 프레임 시간 패널티
- **-100**: 트랙 밖으로 나감 (에피소드 종료)

## 🔧 하이퍼파라미터

```python
TOTAL_TIMESTEPS = 100000    # 총 학습 스텝
LEARNING_RATE = 0.0003      # 학습률
N_STEPS = 2048              # 경험 수집 스텝
BATCH_SIZE = 64             # 배치 크기
N_EPOCHS = 10               # 학습 에포크
```

## 📈 학습 진행 모니터링

TensorBoard로 학습 진행 상황을 실시간으로 확인할 수 있습니다:

```bash
tensorboard --logdir=./logs
```

## 🎥 결과 확인

학습 후 `test_car_racing.py`를 실행하면:
- 3개 에피소드 자동 실행
- `video_output/` 폴더에 MP4 파일 저장
- 각 에피소드의 점수와 스텝 수 출력

## 📦 필요한 패키지

- `gymnasium[box2d]` - CarRacing 환경
- `stable-baselines3[extra]` - PPO 알고리즘
- `moviepy` - 비디오 녹화
- `pygame` - 렌더링

## 🤝 기여

이슈와 Pull Request는 언제나 환영합니다!

## 📄 라이선스

MIT License

## 🔗 참고 자료

- [Gymnasium Documentation](https://gymnasium.farama.org/environments/box2d/car_racing/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

## 💡 팁

### 학습이 잘 안 될 때
- 학습 시간 늘리기 (TOTAL_TIMESTEPS 증가)
- 하이퍼파라미터 조정
- 체크포인트에서 재개하기

### 더 빠른 학습
- GPU 사용 (Google Colab 권장)
- 여러 환경 병렬 실행 (SubprocVecEnv)

---

만든이: Your Name
