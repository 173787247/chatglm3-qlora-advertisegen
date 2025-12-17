# ZeRO-3 配置文件优化说明

## 问题分析

原始配置在训练 T5-3B 时出现 CPU 内存分配失败错误：
```
RuntimeError: DefaultCPUAllocator: can't allocate memory: you tried to allocate 3389157376 bytes. Error code 12 (Cannot allocate memory)
```

## 优化策略

### 1. 减少内存峰值使用

**关键参数调整：**

- `stage3_max_live_parameters`: `1e9` → `1e8` (1亿)
  - 减少同时保存在内存中的参数数量
  - 降低 CPU 内存峰值使用

- `stage3_max_reuse_distance`: `1e9` → `1e8`
  - 减少参数重用距离
  - 更频繁地释放不再使用的参数

- `stage3_param_persistence_threshold`: `"auto"` → `1e5` (10万)
  - 设置参数持久化阈值
  - 更早释放不常用的参数

### 2. 优化 Offload 配置

- `pin_memory`: `true` → `false`
  - 禁用内存锁定，减少内存压力
  - 虽然可能略微降低性能，但能显著减少内存使用

### 3. 调整通信缓冲区大小

- `reduce_bucket_size`: `"auto"` → `5e7` (5000万)
- `stage3_prefetch_bucket_size`: `"auto"` → `5e7`
- `sub_group_size`: `1e9` → `1e8`
  - 设置固定值，避免自动分配过大
  - 减少通信缓冲区内存占用

## 配置文件对比

| 参数 | 原始值 | 优化值 | 说明 |
|------|--------|--------|------|
| `stage3_max_live_parameters` | 1e9 | 1e8 | 减少10倍 |
| `stage3_max_reuse_distance` | 1e9 | 1e8 | 减少10倍 |
| `stage3_param_persistence_threshold` | "auto" | 1e5 | 固定阈值 |
| `pin_memory` (optimizer) | true | false | 减少内存锁定 |
| `pin_memory` (param) | true | false | 减少内存锁定 |
| `reduce_bucket_size` | "auto" | 5e7 | 固定大小 |
| `stage3_prefetch_bucket_size` | "auto" | 5e7 | 固定大小 |
| `sub_group_size` | 1e9 | 1e8 | 减少10倍 |

## 支持的模型

优化后的配置可以支持：
- ✅ T5-3B (google/t5-3b)
- ✅ T5-11B (google/t5-v1_1-xxl)

## 使用方法

### 训练 T5-3B
```bash
bash train_t5_3b_zero3.sh
```

### 训练 T5-11B
```bash
bash train_t5_11b_zero3.sh
```

## 验证方法

训练启动后，检查：
1. **nvidia-smi**: GPU 内存使用情况
2. **htop**: CPU 内存使用情况
3. 训练日志：确认没有内存分配错误

## 注意事项

1. 如果仍然出现内存问题，可以进一步减小：
   - `stage3_max_live_parameters`: 1e7
   - `stage3_max_reuse_distance`: 1e7
   - `reduce_bucket_size`: 1e7

2. 对于 T5-11B 等超大模型，建议：
   - 使用更小的 batch size
   - 增加 gradient_accumulation_steps
   - 考虑使用多 GPU 训练

3. 性能权衡：
   - 较小的参数缓存可能略微降低训练速度
   - 但能确保训练可以正常运行

