# Save the Prey

本课设基于 `multiagent` 追逃仿真框架，包含场景、物理积分、碰撞处理和 RGB 渲染。适配层保留现有的图像特征提取、NumPy MLP 行为克隆、势场安全控制器和追击者控制。猎物策略只读取渲染图像推导出的特征，并输出二维有界加速度。

## 环境安装

建议使用 Python 3.9 至 3.12，并安装锁定依赖：

```bash
python -m pip install -r requirements.txt
```

Pyglet 渲染器通过 OpenGL 提供 RGB 帧，因此训练、评估和动画需要可用的图形桌面。无桌面 Linux 服务器需要配置虚拟显示环境。

## 训练

```bash
python scripts.py train --episodes 120 --seed 2026
```

训练会重新生成 `models/visual_policy.npz` 和 `models/training_metrics.json`。仓库包含的 `models/visual_policy.npz` 可直接用于评估和演示，无需先训练。

## 评估

```bash
python scripts.py evaluate --model models/visual_policy.npz --episodes 30 --seed 5000 --mlp-weight 0.25 --output results/holdout.json
```

添加 `--teacher-baseline` 可只评估势场控制器。指定 `--layouts` 可对 JSON 布局进行评估，输出包括每回合状态、步数和最终坐标。

## 动画演示

```bash
python scripts.py play --layouts configs/official_test_layout.json --mlp-weight 0.25
```

演示只运行正式测试条件这一轮：追击者初始位置 `[0, 0]`，被追者初始位置 `[-0.5, 0.5]`，目标点位置 `[0.5, -0.5]`。窗口提供暂停/继续、重新开始和全屏控制；空格暂停，`R` 重开，`F11` 全屏，`Esc` 退出全屏。可用 `--fps` 和 `--display-size` 调整播放速度和窗口图像尺寸。

## 布局

正式布局保存在 `configs/official_test_layout.json`。其他测试坐标可参照 `configs/official_layouts.example.json`，每个布局需提供追击者、被追者、目标点和三个障碍物的二维坐标，坐标范围为 `[-1, 1]`。

## 仿真实现

`multiagent/` 中包含核心环境文件：`core.py`、`environment.py`、`multi_discrete.py`、`rendering.py`、`scenario.py` 和 `scenarios/simple_tag.py`。策略使用 NumPy MLP 与安全控制器，不依赖 PPO 实现。依赖见 `requirements.txt`。

当前仿真验证：随附模型在正式测试坐标下成功到达目标，用时 98 步；seed 5000 的 30 个随机布局评估成功 26 次（86.7%），平均 48.9 步。对应日志为 `results/official_test_teacher_env.json` 和 `results/teacher_env_holdout_seed5000.json`。其他旧评估日志由 Pillow 简化模拟器生成，属于历史记录，与当前仿真设置不同。

运行回归测试：

```bash
python -m unittest -v test_perception.py test_teacher_environment.py
```
