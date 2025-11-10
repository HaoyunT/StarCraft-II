"""
训练配置快速切换

提供不同的训练配置方案，可根据需要选择
"""

# ============= 经验回放配置 =============
# 配置1: 不使用经验回放（最稳定，推荐用于初次训练）
USE_REPLAY_BUFFER = False

# 配置2: 使用经验回放（提高样本效率，训练稳定后可尝试）
# USE_REPLAY_BUFFER = True

# 注意：经验回放可能因为episode长度不同导致维度不匹配
# 如果遇到 "dimensions must match" 错误，请设置为 False

# 配置3: 探索参数
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY_RATIO = 0.7

# 配置4: 经验回放参数
REPLAY_BUFFER_SIZE_MULTIPLIER = 5  # buffer大小 = n_episodes * 此值
NEW_DATA_RATIO = 0.5  # 新数据占比（0.5表示50%新数据+50%历史数据）

print(f"训练配置:")
print(f"  经验回放: {'启用' if USE_REPLAY_BUFFER else '禁用'}")
print(f"  探索率: {EPSILON_START} -> {EPSILON_END}")
print(f"  探索衰减: {EPSILON_DECAY_RATIO*100}%训练时间")
if USE_REPLAY_BUFFER:
    print(f"  Buffer大小: {REPLAY_BUFFER_SIZE_MULTIPLIER}x批次大小")
    print(f"  新旧数据比: {NEW_DATA_RATIO*100:.0f}%:{(1-NEW_DATA_RATIO)*100:.0f}%")

