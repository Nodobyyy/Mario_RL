# 第三方来源

- 原 Mario PPO / Dreamer 项目与预训练策略：William Kang，https://github.com/WilliamK112/mario-machine-learning 。交付的是原始源码归档。原仓库未见顶层 LICENSE；本项目不为其代码或模型另行授予许可，使用与分发前请遵循原作者条件。
- NES 核心和 Python 包：Christian Kauten，https://github.com/Kautenja/nes-py ，8.2.1，MIT。`vendor/wheels/` 的 Windows wheel 由官方 PyPI sdist 构建；包含其 LICENSE。
- Mario Gym 环境：Christian Kauten，https://github.com/Kautenja/gym-super-mario-bros ，7.4.0；PyPI 声明 Free For Educational Use。游戏及商标的权利属于其各自权利人。
- Stable-Baselines3，https://github.com/DLR-RM/stable-baselines3 ，MIT。
- PyTorch，https://github.com/pytorch/pytorch ，BSD 风格许可。
- React / Vite / TypeScript / Lucide / ECharts：相应包中的原始许可适用。前端构建未使用外部字体或远程图像。

项目不修改上游的版权声明；开发的新适配层不改变第三方的授权条件。依赖安装从官方包源获取，Windows Python 便携运行时取自官方 NuGet Python 包。
