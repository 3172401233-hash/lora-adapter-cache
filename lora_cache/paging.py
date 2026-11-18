"""分页加载与版本一致性。

adapter 按页加载；加载中断（显存回收、取消、抢占）时，
已提交的旧版本保持可用，新版本未提交的部分直接丢弃，
读取方永远只看到完整提交的版本。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .models import AdapterSpec


class LoadStatus(enum.Enum):
    LOADING = "loading"
    COMMITTED = "committed"
    INTERRUPTED = "interrupted"


@dataclass
class _Session:
    spec: AdapterSpec
    num_pages: int
    pages_loaded: int = 0
    status: LoadStatus = LoadStatus.LOADING


@dataclass
class PagedCache:
    """按页提交 adapter 版本的缓存视图。

    - ``committed`` 记录当前对请求可见的版本；
    - 加载中的会话不覆盖已提交版本；
    - ``interrupt`` 丢弃半成品，旧版本原地保留（回退语义）。
    """

    page_size: int = 1 << 20  # 1 MiB / 页
    committed: dict[str, AdapterSpec] = field(default_factory=dict)
    _sessions: dict[str, _Session] = field(default_factory=dict)

    def begin_load(self, spec: AdapterSpec) -> None:
        num_pages = max(1, -(-spec.size // self.page_size))
        self._sessions[spec.adapter_id] = _Session(spec=spec, num_pages=num_pages)

    def load_page(self, adapter_id: str) -> LoadStatus:
        session = self._sessions[adapter_id]
        if session.status is not LoadStatus.LOADING:
            return session.status
        session.pages_loaded += 1
        if session.pages_loaded >= session.num_pages:
            # 全部页就绪后原子提交，读者一次性看到新版本
            session.status = LoadStatus.COMMITTED
            self.committed[adapter_id] = session.spec
        return session.status

    def interrupt(self, adapter_id: str) -> LoadStatus:
        """中断加载：丢弃未提交页，旧版本保持可用。"""
        session = self._sessions.pop(adapter_id, None)
        if session is None or session.status is LoadStatus.COMMITTED:
            return LoadStatus.COMMITTED
        session.status = LoadStatus.INTERRUPTED
        return LoadStatus.INTERRUPTED

    def visible_version(self, adapter_id: str) -> int | None:
        spec = self.committed.get(adapter_id)
        return spec.version if spec else None
