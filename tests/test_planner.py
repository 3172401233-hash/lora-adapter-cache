import unittest

from lora_cache import (
    AdapterSpec,
    BaseModel,
    LoadStatus,
    LoRaCachePlanner,
    PagedCache,
    Request,
    RequestState,
)

MB = 1 << 20
BASE = BaseModel("llama-7b", size=13_000 * MB)


def make_adapters():
    return {
        "chat": AdapterSpec("chat", size=400 * MB),
        "code": AdapterSpec("code", size=400 * MB),
        "math": AdapterSpec("math", size=400 * MB),
        "legal": AdapterSpec("legal", size=800 * MB),
    }


def active(rid, aid, arrival):
    return Request(rid, aid, RequestState.ACTIVE, arrival=arrival)


class ColdStartTest(unittest.TestCase):
    def test_cold_start_loads_required_adapters(self):
        planner = LoRaCachePlanner(total_budget=16_000 * MB)
        reqs = [active("r1", "chat", 0), active("r2", "code", 1)]
        plan = planner.plan(make_adapters(), reqs, BASE, now=0)

        self.assertTrue(plan.feasible)
        self.assertEqual({l.adapter_id for l in plan.loads}, {"chat", "code"})
        self.assertTrue(all(l.reason == "cold_start" for l in plan.loads))
        self.assertEqual(plan.evictions, [])
        self.assertEqual(set(plan.admitted), {"r1", "r2"})
        # 基础模型 + 两个 adapter 都在预算内
        self.assertEqual(plan.usage["inference_used"], 800 * MB)


class ConcurrentSwitchTest(unittest.TestCase):
    def test_concurrent_switch_keeps_active_pinned(self):
        planner = LoRaCachePlanner(total_budget=14_000 * MB,
                                   prefetch_fraction=0.0)
        adapters = make_adapters()
        # 推理池仅 1 GB：chat/code/math 三者不能同时驻留
        reqs = [active("r1", "chat", 0), active("r2", "code", 1)]
        planner.plan(adapters, reqs, BASE, now=0)

        # 并发切换：r1 仍在 chat 上推理，新请求切到 math
        reqs2 = [active("r1", "chat", 0), active("r3", "math", 2)]
        plan = planner.plan(adapters, reqs2, BASE, now=1)

        # chat 被活跃请求 pin 住，绝不淘汰
        self.assertNotIn("chat", {e.adapter_id for e in plan.evictions})
        # code 失去引用，按 LRU 淘汰
        self.assertIn("code", {e.adapter_id for e in plan.evictions})
        self.assertIn("math", {l.adapter_id for l in plan.loads})
        self.assertEqual(set(plan.admitted), {"r1", "r3"})

    def test_interleaved_switch_no_model_flip_for_active(self):
        planner = LoRaCachePlanner(total_budget=14_000 * MB,
                                   prefetch_fraction=0.0)
        adapters = make_adapters()
        reqs = [active("r1", "chat", 0)]
        planner.plan(adapters, reqs, BASE, now=0)
        # r1 活跃期间其它请求到来，r1 的 adapter 必须保持驻留
        for tick in range(1, 4):
            reqs = [active("r1", "chat", 0),
                    Request(f"q{tick}", "code", RequestState.PENDING, tick)]
            plan = planner.plan(adapters, reqs, BASE, now=tick)
            self.assertIn("chat", planner.resident)
            self.assertNotIn("chat", {e.adapter_id for e in plan.evictions})


class HotAdapterTest(unittest.TestCase):
    def test_hot_adapter_survives_lru(self):
        planner = LoRaCachePlanner(total_budget=14_000 * MB,
                                   prefetch_fraction=0.0)
        adapters = make_adapters()
        planner.plan(adapters, [active("r1", "chat", 0),
                                active("r2", "code", 1)], BASE, now=0)
        # chat 持续被引用（热），code 变冷；math 到来时淘汰 code
        plan = planner.plan(adapters, [active("r3", "chat", 2),
                                       active("r4", "math", 3)], BASE, now=1)
        evicted = {e.adapter_id for e in plan.evictions}
        self.assertIn("code", evicted)
        self.assertNotIn("chat", evicted)
        # 热 adapter 不需要重新加载
        self.assertNotIn("chat", {l.adapter_id for l in plan.loads})


class BudgetShortageTest(unittest.TestCase):
    def test_insufficient_budget_defers_and_falls_back(self):
        planner = LoRaCachePlanner(total_budget=14_000 * MB,
                                   prefetch_fraction=0.0)
        adapters = make_adapters()
        # 推理池 1 GB：chat + code + math 共 1.2 GB，装不下
        reqs = [active("r1", "chat", 0), active("r2", "code", 1),
                active("r3", "math", 2)]
        plan = planner.plan(adapters, reqs, BASE, now=0)

        self.assertFalse(plan.feasible)
        self.assertEqual(set(plan.admitted), {"r1", "r2"})
        self.assertEqual(plan.deferred, ["r3"])
        fb = plan.fallbacks[0]
        self.assertEqual(fb.request_id, "r3")
        self.assertEqual(fb.reason, "budget_exhausted")
        # math 从未驻留，回退到基础模型
        self.assertIsNone(fb.keep_adapter_id)

    def test_fallback_keeps_previous_adapter_when_resident(self):
        planner = LoRaCachePlanner(total_budget=14_000 * MB,
                                   prefetch_fraction=0.0)
        adapters = make_adapters()
        planner.plan(adapters, [active("r1", "chat", 0)], BASE, now=0)
        # r2 曾在 chat 上，切换 legal(800MB) 时与 chat(400MB) 超出 1GB 池
        reqs = [active("r1", "chat", 0), active("r2", "legal", 1)]
        plan = planner.plan(adapters, reqs, BASE, now=1)
        self.assertFalse(plan.feasible)
        fb = next(f for f in plan.fallbacks if f.request_id == "r2")
        # legal 未驻留 → 回退基础模型；已准入的 r1 不受影响
        self.assertIsNone(fb.keep_adapter_id)
        self.assertIn("r1", plan.admitted)

    def test_base_model_alone_exceeds_budget(self):
        planner = LoRaCachePlanner(total_budget=1 * MB)
        plan = planner.plan(make_adapters(), [active("r1", "chat", 0)],
                            BASE, now=0)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.loads, [])


class CancelTest(unittest.TestCase):
    def test_cancelled_request_releases_adapter(self):
        planner = LoRaCachePlanner(total_budget=14_000 * MB,
                                   prefetch_fraction=0.0)
        adapters = make_adapters()
        reqs = [active("r1", "chat", 0), active("r2", "code", 1)]
        planner.plan(adapters, reqs, BASE, now=0)

        # r2 取消：code 不再被引用，立即淘汰释放显存
        reqs[1].cancel()
        plan = planner.plan(adapters, reqs, BASE, now=1)
        evicted = {e.adapter_id: e for e in plan.evictions}
        self.assertIn("code", evicted)
        self.assertEqual(evicted["code"].reason, "cancelled")
        self.assertNotIn("code", planner.resident)
        # chat 仍被 r1 引用，保持驻留
        self.assertIn("chat", planner.resident)

    def test_cancelled_shared_adapter_stays_resident(self):
        planner = LoRaCachePlanner(total_budget=14_000 * MB,
                                   prefetch_fraction=0.0)
        adapters = make_adapters()
        reqs = [active("r1", "chat", 0), active("r2", "chat", 1)]
        planner.plan(adapters, reqs, BASE, now=0)
        reqs[1].cancel()
        plan = planner.plan(adapters, reqs, BASE, now=1)
        # r1 仍活跃，chat 不因 r2 取消而淘汰
        self.assertNotIn("chat", {e.adapter_id for e in plan.evictions})
        self.assertIn("chat", planner.resident)


class PrefetchTest(unittest.TestCase):
    def test_pending_requests_prefetch_within_pool(self):
        planner = LoRaCachePlanner(total_budget=15_000 * MB,
                                   prefetch_fraction=0.25)
        adapters = make_adapters()
        reqs = [active("r1", "chat", 0),
                Request("q1", "math", RequestState.PENDING, 1)]
        plan = planner.plan(adapters, reqs, BASE, now=0)
        self.assertIn("math", {p.adapter_id for p in plan.prefetches})
        # 预取条目可被活跃需求挤掉
        self.assertTrue(planner.resident["math"].prefetch)


class PagedInterruptTest(unittest.TestCase):
    def test_interrupt_keeps_old_version_visible(self):
        cache = PagedCache(page_size=100 * MB)
        old = AdapterSpec("chat", size=200 * MB, version=1)
        cache.begin_load(old)
        while cache.load_page("chat") is LoadStatus.LOADING:
            pass
        self.assertEqual(cache.visible_version("chat"), 1)

        # 版本升级加载到一半被中断：旧版本必须保持可用
        new = AdapterSpec("chat", size=200 * MB, version=2)
        cache.begin_load(new)
        cache.load_page("chat")
        status = cache.interrupt("chat")
        self.assertEqual(status, LoadStatus.INTERRUPTED)
        self.assertEqual(cache.visible_version("chat"), 1)

    def test_commit_is_atomic(self):
        cache = PagedCache(page_size=100 * MB)
        spec = AdapterSpec("code", size=300 * MB, version=1)
        cache.begin_load(spec)
        cache.load_page("code")
        cache.load_page("code")
        # 未加载完，读者看不到任何版本
        self.assertIsNone(cache.visible_version("code"))
        cache.load_page("code")
        self.assertEqual(cache.visible_version("code"), 1)


if __name__ == "__main__":
    unittest.main()
