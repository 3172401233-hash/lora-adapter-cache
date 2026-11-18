# lora-adapter-cache

Python LoRA Adapter 缓存库，处理基础模型共享、页预算、并发请求和淘汰回退。

多个推理请求加载不同 LoRA 权重时，基础模型只有一份常驻显存，
adapter 与 KV Cache 竞争同一份显存预算。本库实现一个缓存规划器：
输入 adapter 目录（ID / 大小 / 版本）、引用请求、基础模型、
最近使用顺序和显存预算，输出**加载、淘汰、回退**计划。

## 结构

- `lora_cache/models.py` — 输入模型：`AdapterSpec`、`BaseModel`、`Request`
- `lora_cache/planner.py` — `LoRaCachePlanner.plan()`，输出 `Plan`
- `lora_cache/paging.py` — `PagedCache`，分页加载与原子版本提交
- `tests/test_planner.py` — 并发切换、热 adapter、冷启动、预算不足、取消请求

## 预算划分

```
总预算 = 基础模型 + adapter 预算
adapter 预算 = 推理池 (1 - prefetch_fraction) + 预取池 prefetch_fraction
```

- **推理池**：活跃请求 pin 住的 adapter，绝不淘汰；装不下时按到达顺序准入，
  其余请求延后并生成回退动作。
- **预取池**：排队请求引用的 adapter  opportunistic 预取，
  活跃需求到来时最先让位（`prefetch_shrink`），其余按 LRU 淘汰。

## 模型一致性

- 活跃请求在准入时绑定 adapter 版本（`bound_version`），
  其 adapter 在请求存活期间不会被淘汰，推理结果不会中途切换模型。
- 版本升级通过分页加载进行：新版本的页全部就绪后才原子提交
  （`PagedCache`）；加载中断时丢弃半成品页，旧版本原地保留，
  进行中的请求继续读到旧版本——这就是回退语义。
- 预算不足时，无法准入的请求回退到其之前绑定的 adapter
  （若仍驻留），否则回退到纯基础模型。

## 缓存边界

- 基础模型始终常驻，不参与淘汰，也不计入 adapter 预算。
- 取消的请求立即失去引用；无其他引用的 adapter 马上进入淘汰集合
  （`reason="cancelled"`），共享 adapter 则不受影响。
- KV Cache 的显存压力通过 `total_budget` 收紧体现：
  规划器只保证 adapter 驻留不超过推理池 + 预取池。
- 预取是尽力而为：预取池装满即停止，不侵占推理池。

## 运行测试

```bash
python -m unittest
```
