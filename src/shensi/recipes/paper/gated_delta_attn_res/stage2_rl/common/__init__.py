"""强化学习段的公共件：四方向 teacher 共用的启动、数据准备与配置装载。"""

from .launch import main as launch_main
from .prep import prepare

__all__ = ["launch_main", "prepare"]
