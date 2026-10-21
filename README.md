# Save the Prey

本课设实现一个基于图像的猎物控制策略：策略读取 800×800 RGB 图像并输出二维有界加速度。感知使用 RGB 颜色分割；机器学习部分使用 NumPy MLP 进行行为克隆；运行时还将 MLP 与势场安全控制器融合。捕食者使用带障碍物排斥的追踪策略，模拟器同时约束其与障碍物保持接触距离，避免穿透。

独立环境保留课设中的固定障碍物位置、画面大小、速度和加速度限制、捕获规则及 shelter 规则。感知同时支持本地颜色和原 Pyglet 渲染器中 alpha 混合后的运动体颜色。每个坐标轴每隔 4 个像素采样一次；在 Python 3.12.14、NumPy 2.3.5、Pillow 12.3.0 上，本机基准测试的单帧耗时为 5.38 ms（预热 30 次后测量 500 次）。障碍物像素过少时，感知会返回近似或重复的中心以避免评估崩溃；这些值不代表真实恢复出的障碍物位置。

本地环境在边界和 shelter 形状上与原环境仍有差异：本地在坐标达到 0.975 时终止，原环境约从 0.825 开始施加边界接触力；本地 shelter 为圆形，原环境中的 check 是小方块。Pillow 使项目无需 Pyglet、OpenGL、Gymnasium 或图形桌面，可在 Windows 和无桌面的 Ubuntu 上运行。运行时策略只接收图像推导出的特征，不读取环境内部位置；模拟器和捕食者控制器可以使用内部状态。

## 环境安装

建议使用 Python 3.9 或更高版本，并安装项目锁定的 NumPy、Pillow 版本：

```bash
python -m pip install -r requirements.txt
```

## 训练

在项目目录运行：

```bash
python scripts.py train --episodes 120 --seed 2026
```

仓库附有正式模型 `models/visual_policy.npz`，可直接进行评估和动画演示。执行训练命令会重新生成模型权重和 `models/training_metrics.json`。当前模型使用修正后的全分辨率感知管线训练；之后加入的 4×点采样优化没有触发重新训练。模型文件保存 NumPy 权重及训练、环境配置。增加 `--episodes` 可扩大示范数据；调整 `--epochs` 可改变 MLP 训练轮数。

`models/visual_policy_candidate_*.npz` 是后续实验得到的候选模型，尚未达到目标成功率，因此不是默认交付模型。正式模型仍为 `models/visual_policy.npz`；候选模型的训练和评估记录在 `results/training_candidate_*.json`。

## 随机留出评估

```bash
python scripts.py evaluate --model models/visual_policy.npz --episodes 30 --seed 5000 --mlp-weight 0.25 --output results/holdout.json
```

评估记录包含成功、被捕获、障碍物碰撞、越界、超时、平均步数和逐回合最终状态。添加 `--teacher-baseline` 可只评估势场教师。

## 动画演示

在 Windows 桌面环境中播放第五组测试条件：

```bash
python scripts.py play --layouts configs/five_shelter_demo.json --mlp-weight 0.25
```

该布局固定追击者初始位置 `[0, 0]`、被追者初始位置 `[-0.5, 0.5]`、目标点 `[0.5, -0.5]`。

如需录制不同 shelter 位置，可使用固定的随机布局：

```bash
python scripts.py play --layouts configs/random_shelter_demo_24.json --mlp-weight 0.25 --fps 15
```

共 24 个不同的 shelter 位置。这组位置与初始猎物至少相距 0.5，并避开障碍物和捕食者附近，减少开场很快结束或目标紧贴障碍的情况。`--fps 15` 会放慢动画。窗口左侧显示当前轮数；每轮结束后会停留在结果画面，点击“下一布局”或按右方向键后才开始下一轮。每次运行使用相同的位置，便于重复录制。

播放正式布局时，将正式坐标保存为 `configs/official_layouts.json`，再运行：

```bash
python scripts.py play --layouts configs/official_layouts.json --mlp-weight 0.25
```

窗口默认以 600 像素画面显示，确保控制按钮可见。窗口提供暂停/继续、重新开始、切换布局控制；空格暂停，`R` 重开，右方向键切换布局，`F11` 全屏，`Esc` 退出全屏。`--fps` 可调整播放速度，例如 `--fps 20`；`--display-size` 可调整画面尺寸。动画使用项目的 Pillow 独立环境，不是原始 Pyglet 渲染器。

布局读取器要求坐标为 `[-1, 1]` 内的有限二维数值，每个布局恰有 3 个有限障碍物坐标，且猎物初始位置不能与捕食者或障碍物重叠。若近距离障碍物在 RGB 图像中连成一片，感知会聚类估计三个中心；可见障碍像素少于 20 时，会复制可见中心，完全不可见时会使用中性占位中心。此时策略的避障信息已降级，但评估流程仍可继续。

## 正式布局

`configs/official_layouts.example.json` 使用指定初始坐标：猎物 `[-0.5, 0.5]`、猎手 `[0, 0]`、目标 `[0.5, -0.5]`。如需使用老师公布的其他坐标，复制该文件并替换坐标，保存为 `configs/official_layouts.json`。坐标使用 `[-1, 1]` 范围内的世界坐标，每个布局包含三个障碍物位置。运行：

```bash
python scripts.py evaluate --model models/visual_policy.npz --layouts configs/official_layouts.json --mlp-weight 0.25 --output results/official.json
```

## Ubuntu 运行与验证

将本目录（包括 `models/visual_policy.npz` 和布局 JSON）复制到 Ubuntu，安装 `requirements.txt` 中指定的依赖后即可运行相同的评估命令。无需图形桌面或在 Ubuntu 上重新训练。

本交付包已在 WSL2 Ubuntu 上验证，运行环境为 Python 3.12.3、NumPy 1.26.4、Pillow 10.4.0。7 项感知测试全部通过；示例布局评估为 1/1 成功、61 步。seed 5000 的 30 布局评估及 seed 777 的 200 布局评估均已运行。200 布局的教师、融合策略、纯 MLP 分别为 172/200、172/200、126/200，与保存日志一致。此处验证的是 WSL2 Linux 环境，不等同于老师的正式评估环境。

## 当前随机留出结果

seed 5000 的一组 30 布局日志中，教师和权重为 0.25 的融合策略均成功 27/30（90.0%）；教师有 2 次捕获和 1 次障碍物碰撞，平均 53.1 步；融合策略有 2 次捕获和 1 次障碍物碰撞，平均 51.0 步。对应记录为 `results/teacher_holdout.json` 和 `results/holdout_weight_025.json`。

独立权重扫描在权重 0.00、0.10、0.25、0.40、0.55、0.70、0.85、1.00 下分别成功 27、27、27、25、24、21、21、19 次。这个单种子扫描没有显示权重不超过 0.25 时成功数量提升；较高权重的成功数较低。它不代表普遍的权重响应规律。纯 MLP 的单独留出日志记录为 18/30；扫描中的权重 1.00 是另一轮运行，记录为 19/30。各轮数据见 `results/holdout_weight_100.json` 和 `results/weight_scan_*.json`。

在 seed 777 的 200 个独立布局上，教师成功 172/200（86.0%），有 17 次捕获、7 次障碍物碰撞、4 次超时，平均 71.8 步；融合策略成功 172/200（86.0%），有 15 次捕获、8 次障碍物碰撞、5 次超时，平均 76.0 步；纯 MLP 成功 126/200（63.0%），有 22 次捕获、44 次障碍物碰撞、1 次越界、7 次超时，平均 79.4 步。逐回合记录分别见 `results/neutral_200_teacher.json`、`results/neutral_200_weight_025.json` 和 `results/neutral_200_weight_100.json`。

运行感知和布局输入回归测试：

```bash
python -m unittest -v test_perception.py
```

`results/windows_review.json` 保存了 Windows 上示例布局的测试记录：成功，61 步，无捕获、障碍物碰撞或越界事件。
