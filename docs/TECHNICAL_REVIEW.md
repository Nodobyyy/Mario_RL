# 上游技术审查与实现决策

审查日期：2026-10-27。来源为指定 GitHub 仓库 main 源码归档，未改写其源文件。

## 可复用部分

- `mario_runtime.py`：EnvConfig、JoypadSpace、跳帧、灰度缩放、VecFrameStack、奖励函数、flag_get 成功处理、坐标清洗与结果排序。
- `train_mario.py`：PPO 创建 / load、跳跃 logit 先验、VecNormalize、原始 `model.learn` 训练流程与最终保存。
- `evaluate_mario.py` / `render_best_checkpoint.py`：四帧堆叠推理和视频导出的实现方式。Dashboard 独立评估采用相同预处理，增加逐步遥测和流式视频写入，避免把长录像全部保留在内存。
- `pretrained/ppo_mario_1-1/`：原始 best_eval.zip、环境配置与 vecnormalize.pkl。

## 依赖表与实际内容不一致

1. 上游 import / requirements 使用未能在 PyPI 找到的 `gymnasium-super-mario-bros==7.4.0`。实际采用官方 `gym-super-mario-bros==7.4.0` + `nes-py==8.2.1`，在 `backend/training/compat.py` 桥接命名空间 / Gymnasium spaces；由上游包装器完成旧式 step/reset 转换。
2. 上游 requirements 锁定 SB3 2.3.2、NumPy 1.23.5；但交付的权重 `system_info.txt` 明确写 SB3 2.7.0、Python 3.12.10、PyTorch 2.11.0+cu128、NumPy 2.4.4、Gymnasium 1.2.1。
3. 当前训练兼容配置使用 Python 3.11、Torch 2.4.1、SB3 2.7.0、NumPy 1.26.4、Gymnasium 0.29.1。载入时重建同形状 spaces、丢弃不应跨模拟器恢复的 last_obs / episode starts；策略和 Adam 参数从原始独立 tensor 文件加载，不更改权重。
4. 上游训练文档中的“约 8.45M”不是 best_eval 的精确累计值。权重内 `num_timesteps` 实测 **8,050,000**，系统从文件中读取它，不使用宣传描述估算。
5. Windows 官方 nes-py 8.2.1 没有对应二进制 wheel，pip 常要求 MSVC。项目通过 Zig 0.13.0 编译官方 sdist 的 ctypes C++ 核心为便携 DLL，并生成平台 wheel。安装器优先使用交付的 wheel。
6. 上游依赖同时包含 DreamerV3、MuJoCo、dm_control；本阶段不安装这些依赖。

原版 VecNormalize pickle 同样使用 NumPy 2.4 的五参数 `_frombuffer`。兼容读取器按该版本的维度 / 轴顺序恢复完整数组和 running mean / variance，使用新模拟器自己的 spaces / RNG。转换规则核对了 [NumPy 2.4.4 官方 numeric.py](https://github.com/numpy/numpy/blob/v2.4.4/numpy/_core/numeric.py)。原预训练 reward RMS count 实测为 8,507,392.0001；CPU 微调已通过端到端验收。

## 新增适配层

训练器通过模块的 factory / callback / parse_args 接入原 `main()`，没有复制 PPO 的优化算法。PPO seed 注入原构造器；从 checkpoint 恢复时载入其环境配置，保留动作维度、观测形状和奖励配方。

采集 wrapper 位于原环境 Monitor / WarpFrame 外层、奖励归一化之前。完整逐步轨迹写 NPZ，摘要写 SQLite；真实 RGB 帧由环境拥有者提取，后台单元素队列编码 / 写盘，既不读取 checkpoint 冒充训练画面，也不等待客户端消费。

自动评估使用训练边界的不可变 checkpoint，在单独进程、单独环境运行。评估回合的 kind 为 evaluation。最佳选择采用与上游一致的成功率、剩余距离与奖励排序。

恢复的 checkpoint 目录同时保存 VecNormalize 和环境配置。默认禁用上游无法完整恢复状态的 RND，保证纯 PPO 基线的状态可解释；不是声明支持恢复 RND 世界状态或模拟器瞬间状态。

CPU 默认 1 env / 128 rollout / 64 batch / 至多 4 线程；CUDA 默认 8 env / 512 rollout / 256 batch。始终保持 CnnPolicy 和同样的图像 / 动作定义，权重可以跨设备加载。
