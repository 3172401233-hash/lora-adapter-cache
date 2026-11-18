"""LoRA adapter 缓存规划器。

在基础模型共享的前提下，于活跃推理与预取之间划分显存预算，
输出加载 / 淘汰 / 回退计划，并保证活跃 adapter 不被释放、
分页加载中断时旧版本仍然可用。
"""

from .models import (
    AdapterSpec,
    BaseModel,
    Request,
    RequestState,
)
from .planner import (
    EvictAction,
    FallbackAction,
    LoadAction,
    LoRaCachePlanner,
    Plan,
    PrefetchAction,
)
from .paging import LoadStatus, PagedCache

__all__ = [
    "AdapterSpec",
    "BaseModel",
    "Request",
    "RequestState",
    "EvictAction",
    "FallbackAction",
    "LoadAction",
    "LoRaCachePlanner",
    "Plan",
    "PrefetchAction",
    "LoadStatus",
    "PagedCache",
]
