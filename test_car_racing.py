"""
학습된 CarRacing AI 테스트 스크립트
학습된 모델로 실제 주행을 테스트하고 영상을 저장
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import glob

# ==========================================
# 1. 설정
# ==========================================

MODEL_PATH = "./models/final_car_racing_driver"
VIDEO_FOLDER = "./video_output"
NUM_EPISODES = 3  # 테스트할 에피소드 수

# 디렉토리 생성
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# ==========================================
# 2. 모델 로드
# ==========================================

print("🤖 학습된 모델 로딩 중...")

try:
    model = PPO.load(MODEL_PATH)
    print("✅ 모델 로드 완료!")
except FileNotFoundError:
    print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    print("먼저 train_car_racing.py를 실행하여 모델을 학습시켜주세요.")
    exit(1)

# ==========================================
# 3. 환경 설정 (비디오 녹화 포함)
# ==========================================

print("\n🎥 비디오 녹화 환경 설정 중...")

# 녹화 환경 생성
eval_env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",  # 비디오 녹화용
    continuous=True
)

# 모든 에피소드 녹화
eval_env = RecordVideo(
    eval_env,
    video_folder=VIDEO_FOLDER,
    episode_trigger=lambda x: True  # 모든 에피소드 녹화
)

print("✅ 환경 설정 완료!")

# ==========================================
# 4. 주행 테스트
# ==========================================

print("\n" + "="*50)
print("🏁 주행 테스트 시작!")
print("="*50)

for episode in range(NUM_EPISODES):
    print(f"\n📍 에피소드 {episode + 1}/{NUM_EPISODES} 시작...")

    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # 모델이 행동 예측
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"🏁 에피소드 {episode + 1} 종료!")
    print(f"   총 스텝: {steps}")
    print(f"   획득 점수: {total_reward:.2f}")

# 환경 종료
eval_env.close()

# ==========================================
# 5. 저장된 비디오 확인
# ==========================================

print("\n" + "="*50)
print("📹 저장된 비디오 파일:")
print("="*50)

video_files = glob.glob(f'{VIDEO_FOLDER}/*.mp4')
for i, video_file in enumerate(video_files, 1):
    file_size = os.path.getsize(video_file) / (1024 * 1024)  # MB
    print(f"{i}. {os.path.basename(video_file)} ({file_size:.2f} MB)")

print(f"\n✅ 총 {len(video_files)}개의 비디오가 '{VIDEO_FOLDER}' 폴더에 저장되었습니다!")
print("\n💡 비디오를 재생하려면 VLC 등의 플레이어로 열어보세요.")
