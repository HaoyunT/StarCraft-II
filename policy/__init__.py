"""
多智能体强化学习策略模块

包含MAPPO算法的实现（已移除IPPO依赖，保持向后兼容）
"""

from .mappo import MAPPO

# 兼容性：如果需要添加 IPPO，请在 policy/ippo.py 中实现并导入
__all__ = ['MAPPO']
