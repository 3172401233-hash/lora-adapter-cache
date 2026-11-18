"""规划器输入的数据模型。"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class RequestState(enum.Enum):
    ACTIVE = "active"        # 正在推理，pin 住其 adapter
    PENDING = "pending"      # 排队中，可作为预取候选
    CANCELLED = "cancelled"  # 已取消，应立即释放引用


@dataclass(frozen=True)
class AdapterSpec:
    """一个 LoRA adapter 的静态描述。"""

    adapter_id: str
    size: int          # 字节
    version: int = 1   # 权重版本；同名 adapter 换版本时旧版本需保持可用


@dataclass(frozen=True)
class BaseModel:
    """共享的基础模型，常驻显存。"""

    model_id: str
    size: int


@dataclass
class Request:
    """引用某个 adapter 的推理请求。"""

    request_id: str
    adapter_id: str
    state: RequestState = RequestState.PENDING
    arrival: int = 0   # 到达顺序，预算不足时按序准入
    bound_version: int | None = None  # 准入时绑定的 adapter 版本

    def cancel(self) -> None:
        self.state = RequestState.CANCELLED
