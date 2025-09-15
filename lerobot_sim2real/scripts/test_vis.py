import gymnasium as gym
import mani_skill.envs

# 确保 SAPIEN 使用兼容性好的 OpenGL
import os
os.environ.setdefault("SAPIEN_DISABLE_VULKAN", "1")

def main():
    env = gym.make("PushCube-v1", render_mode="human")
    print("环境创建成功，正在重置...")
    obs, _ = env.reset()
    print("环境已重置，开始渲染循环...")
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render() # 这一行负责更新画面
        if (i+1) % 50 == 0:
            print(f"已渲染 {i+1} 帧")
    env.close()
    print("测试完成。")

if __name__ == "__main__":
    main()