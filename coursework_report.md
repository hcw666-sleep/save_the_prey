# Save the Prey 课程设计报告

## 任务与方法

策略根据老师环境渲染的 800×800 RGB 图像控制被追者，使其避开追击者和固定障碍物并到达目标点。图像感知通过颜色分割获取目标相对位置、三个障碍物中心、被追者速度和边界信息，形成 16 维特征。NumPy MLP 通过行为克隆拟合势场控制器的动作，运行时以 0.25 权重融合 MLP 输出和安全控制器动作。策略只使用图像推导的特征；场景和追击者控制器可以访问环境状态。

## 环境实现

仿真使用老师提供的 `multiagent/core.py`、`multiagent/environment.py`、`multiagent/scenarios/simple_tag.py` 和 `multiagent/rendering.py`。场景文件定义追击者、被追者、障碍物、目标点及边界；`core.py` 执行物理积分和接触响应；Pyglet 渲染器提供 RGB 图像。项目适配层把策略的二维连续加速度送入老师环境，并维持既有追击者控制器、障碍物规避、速度限制及回合终止规则。环境所需 Python 包列于 `requirements.txt`。

正式测试坐标为追击者 `[0, 0]`、被追者 `[-0.5, 0.5]`、目标点 `[0.5, -0.5]`，布局位于 `configs/official_test_layout.json`。固定障碍物坐标为 `[-0.35, 0.35]`、`[0.35, 0.35]` 和 `[0, -0.35]`。

## 模型与训练

模型使用 16 个输入特征、两个 64 单元隐藏层及二维输出。训练设置为随机种子 2026、120 个示范回合、Adam 优化器、学习率 0.002、160 轮和批量大小 256。随附模型为 `models/visual_policy.npz`。该模型最初在先前的独立模拟器中训练，现已在老师环境的正式测试布局上进行迁移验证；本次未在老师环境中重新训练。

## 正式条件验证

使用随附模型、MLP 融合权重 0.25，在上述正式初始坐标下运行老师环境：

| 结果 | 数值 |
|---|---:|
| 到达目标点 | 是 |
| 步数 | 98 |
| 被捕获 | 否 |
| 障碍物碰撞 | 否 |
| 越界 | 否 |

逐回合数据保存在 `results/official_test_teacher_env.json`。11 项感知与老师环境集成测试通过，命令为：

```bash
python -m unittest -v test_perception.py test_teacher_environment.py
```

另对随机种子 5000 的 30 个随机布局运行了融合策略评估：成功 26/30（86.7%）、被捕获 3 次、障碍物碰撞 1 次、越界 0 次、超时 0 次，平均 48.9 步。结果保存在 `results/teacher_env_holdout_seed5000.json`。这组新统计由老师环境生成，与下方旧模拟器记录分开。

## 历史评估记录

`results/` 中除 `official_test_teacher_env.json` 和 `teacher_env_holdout_seed5000.json` 外的现有随机留出记录是在旧 Pillow 独立模拟器中生成的。该模拟器与老师环境在物理碰撞、边界和目标点几何方面不同，因此这些数据仅作为旧版本记录，不能作为本环境的效果结论。正式成绩应基于老师公布的评估布局，在当前环境重新运行 `scripts.py evaluate` 后统计。

## 运行

在具有可用 OpenGL 图形桌面的 Python 3.9 至 3.12 环境中安装依赖并播放正式演示：

```bash
python -m pip install -r requirements.txt
python scripts.py play --layouts configs/official_test_layout.json --mlp-weight 0.25
```

训练：

```bash
python scripts.py train --episodes 120 --seed 2026
```

正式条件评估：

```bash
python scripts.py evaluate --model models/visual_policy.npz --layouts configs/official_test_layout.json --mlp-weight 0.25 --output results/official.json
```
