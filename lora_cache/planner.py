"""LoRA 缓存规划器。

输入 adapter 目录、请求集合、基础模型与显存预算，
在活跃推理与预取之间分配空间，输出加载 / 淘汰 / 回退计划。

不变量：
- 活跃请求 pin 住的 adapter 永不淘汰；
- 取消的请求立即失去引用，其 adapter 变为可淘汰；
- 版本升级加载完成前，旧版本对进行中的请求保持可见（回退）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import AdapterSpec, BaseModel, Request, RequestState


@dataclass
class LoadAction:
    adapter_id: str
    size: int
    version: int
    reason: str  # "cold_start" | "demand" | "version_upgrade"


@dataclass
class EvictAction:
    adapter_id: str
    size: int
    reason: str  # "lru" | "prefetch_shrink" | "cancelled"


@dataclass
class PrefetchAction:
    adapter_id: str
    size: int


@dataclass
class FallbackAction:
    request_id: str
    keep_adapter_id: str | None  # 继续使用的旧 adapter；None 表示留在基础模型
    reason: str                  # "budget_exhausted" | "load_interrupted"


@dataclass
class CacheEntry:
    spec: AdapterSpec
    last_used: int
    prefetch: bool = False


@dataclass
class Plan:
    loads: list[LoadAction] = field(default_factory=list)
    evictions: list[EvictAction] = field(default_factory=list)
    prefetches: list[PrefetchAction] = field(default_factory=list)
    fallbacks: list[FallbackAction] = field(default_factory=list)
    admitted: list[str] = field(default_factory=list)  # request_id
    deferred: list[str] = field(default_factory=list)
    feasible: bool = True
    usage: dict[str, int] = field(default_factory=dict)


class LoRaCachePlanner:
    """在固定显存预算内规划 adapter 驻留。"""

    def __init__(self, total_budget: int, prefetch_fraction: float = 0.25) -> None:
        if not 0.0 <= prefetch_fraction <= 1.0:
            raise ValueError("prefetch_fraction 必须在 [0, 1]")
        self.total_budget = total_budget
        self.prefetch_fraction = prefetch_fraction
        self.resident: dict[str, CacheEntry] = {}

    def plan(
        self,
        adapters: dict[str, AdapterSpec],
        requests: list[Request],
        base_model: BaseModel,
        now: int = 0,
    ) -> Plan:
        plan = Plan()
        adapter_budget = self.total_budget - base_model.size
        if adapter_budget < 0:
            plan.feasible = False
            plan.fallbacks.extend(
                FallbackAction(r.request_id, None, "budget_exhausted")
                for r in requests
                if r.state is RequestState.ACTIVE
            )
            plan.deferred.extend(r.request_id for r in requests)
            return plan

        prefetch_pool = int(adapter_budget * self.prefetch_fraction)
        inference_pool = adapter_budget - prefetch_pool

        # 取消的请求不再构成引用；独占的 adapter 立即进入可淘汰集合。
        live = [r for r in requests if r.state is not RequestState.CANCELLED]
        cancelled_ids = {r.adapter_id for r in requests
                         if r.state is RequestState.CANCELLED}
        referenced = {r.adapter_id for r in live}
        for aid in sorted(cancelled_ids - referenced):
            entry = self.resident.pop(aid, None)
            if entry is not None:
                plan.evictions.append(
                    EvictAction(aid, entry.spec.size, "cancelled"))

        # 按到达顺序准入活跃请求；预算装不下的延后并回退。
        active = sorted(
            (r for r in live if r.state is RequestState.ACTIVE),
            key=lambda r: r.arrival,
        )
        required: dict[str, AdapterSpec] = {}
        used = 0
        for req in active:
            spec = adapters[req.adapter_id]
            need = 0 if req.adapter_id in required else spec.size
            if used + need > inference_pool:
                plan.feasible = False
                plan.deferred.append(req.request_id)
                # 回退：保持请求之前绑定的旧 adapter（若仍驻留），否则留在基础模型
                keep = req.adapter_id if req.adapter_id in self.resident else None
                plan.fallbacks.append(
                    FallbackAction(req.request_id, keep, "budget_exhausted"))
                continue
            required[req.adapter_id] = spec
            used += need
            plan.admitted.append(req.request_id)
            req.bound_version = spec.version

        # 淘汰：预取条目先让位，其余按 LRU；活跃 pin 住的绝不淘汰。
        inf_used = used       # 必需 adapter 必然驻留
        pre_used = 0
        for aid, entry in self.resident.items():
            if aid in required:
                continue
            if entry.prefetch:
                pre_used += entry.spec.size
            else:
                inf_used += entry.spec.size
        overflow = [e for aid, e in self.resident.items()
                    if aid not in required]
        overflow.sort(key=lambda e: (not e.prefetch, e.last_used))
        for entry in overflow:
            if inf_used <= inference_pool and pre_used <= prefetch_pool:
                break
            del self.resident[entry.spec.adapter_id]
            reason = "prefetch_shrink" if entry.prefetch else "lru"
            plan.evictions.append(
                EvictAction(entry.spec.adapter_id, entry.spec.size, reason))
            if entry.prefetch:
                pre_used -= entry.spec.size
            else:
                inf_used -= entry.spec.size

        # 加载：缺失或版本过期的必需 adapter。
        cold_start = not self.resident
        for aid, spec in required.items():
            entry = self.resident.get(aid)
            if entry is None:
                reason = "cold_start" if cold_start else "demand"
                plan.loads.append(
                    LoadAction(aid, spec.size, spec.version, reason))
                self.resident[aid] = CacheEntry(spec, last_used=now)
            elif entry.spec.version < spec.version:
                # 旧版本在分页提交前保持可见，见 paging.PagedCache
                plan.loads.append(
                    LoadAction(aid, spec.size, spec.version, "version_upgrade"))
                entry.spec = spec
                entry.last_used = now
            else:
                entry.last_used = now

        # 预取：排队请求引用的 adapter，装进预取池，可被淘汰。
        pending_ids = []
        for r in live:
            if r.state is RequestState.PENDING and r.adapter_id not in required:
                if r.adapter_id not in pending_ids:
                    pending_ids.append(r.adapter_id)
        prefetch_used = sum(
            e.spec.size for e in self.resident.values() if e.prefetch)
        for aid in pending_ids:
            if aid in self.resident:
                continue
            spec = adapters[aid]
            if prefetch_used + spec.size > prefetch_pool:
                continue
            plan.prefetches.append(PrefetchAction(aid, spec.size))
            self.resident[aid] = CacheEntry(spec, last_used=now, prefetch=True)
            prefetch_used += spec.size

        plan.usage = {
            "base_model": base_model.size,
            "inference_pool": inference_pool,
            "inference_used": sum(
                e.spec.size for e in self.resident.values() if not e.prefetch),
            "prefetch_pool": prefetch_pool,
            "prefetch_used": prefetch_used,
            "total_budget": self.total_budget,
        }
        return plan
