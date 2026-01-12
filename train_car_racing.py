"""
CarRacing AI 학습 스크립트
PPO 알고리즘으로 자율주행 AI 훈련
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ==========================================
# 1. 설정
# ==========================================

# 학습 설정
TOTAL_TIMESTEPS = 100000  # 총 학습 스텝 (10만 번 추천)
CHECKPOINT_FREQ = 20000   # 체크포인트 저장 빈도
MODEL_NAME = "car_racing_driver"

# PPO 하이퍼파라미터
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10

# 디렉토리 설정
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
MODEL_DIR = "./models"

# 디렉토리 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. 환경 설정
# ==========================================

print("🏎️ CarRacing 환경 초기화 중...")

# CarRacing-v3 환경 생성
# continuous=True: 부드러운 핸들 조작 (False면 이산적)
env = gym.make("CarRacing-v3", continuous=True)
env = DummyVecEnv([lambda: env])  # 벡터 환경으로 래핑

print("✅ 환경 초기화 완료!")

# ==========================================
# 3. 모델 생성
# ==========================================

print("\n🤖 PPO 모델 생성 중...")

model = PPO(
    "CnnPolicy",                    # CNN 정책 (이미지 입력용)
    env,
    verbose=1,                      # 학습 진행 상황 출력
    tensorboard_log=LOG_DIR,        # TensorBoard 로그 저장
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS
)

print("✅ 모델 생성 완료!")

# ==========================================
# 4. 체크포인트 콜백 설정
# ==========================================

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix=MODEL_NAME
)

# ==========================================
# 5. 학습 시작
# ==========================================

print("\n" + "="*50)
print("🏁 훈련 시작!")
print("="*50)
print(f"총 학습 스텝: {TOTAL_TIMESTEPS:,}")
print(f"체크포인트 저장 빈도: 매 {CHECKPOINT_FREQ:,} 스텝")
print(f"\n목표: 점수가 0점을 넘어 양수가 되면 운전을 시작한 것입니다!")
print("=" * 50 + "\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback
    )

    print("\n" + "="*50)
    print("🎉 훈련 완료!")
    print("="*50)

except KeyboardInterrupt:
    print("\n\n⚠️ 훈련이 사용자에 의해 중단되었습니다.")

# ==========================================
# 6. 최종 모델 저장
# ==========================================

final_model_path = os.path.join(MODEL_DIR, f"final_{MODEL_NAME}")
model.save(final_model_path)
print(f"\n💾 최종 모델 저장 완료: {final_model_path}")

# 환경 종료
env.close()
print("\n✅ 모든 작업 완료!")
