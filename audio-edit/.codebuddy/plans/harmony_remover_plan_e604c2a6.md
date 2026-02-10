---
name: harmony_remover_plan
overview: 创建基于频谱掩码的和音消除应用，复用 singer_cleaner.py 界面框架，实现从人声干音中提取目标主唱的功能。
todos:
  - id: create-harmony-remover
    content: 创建 harmony_remover.py，复用 singer_cleaner.py 的界面架构和 WaveformWidget
    status: completed
  - id: implement-stft-istft
    content: 实现 STFT/ISTFT 音频处理核心算法和频谱特征提取
    status: completed
    dependencies:
      - create-harmony-remover
  - id: implement-soft-mask
    content: 实现基于余弦相似度的软掩码生成算法
    status: completed
    dependencies:
      - implement-stft-istft
  - id: integrate-ui-logic
    content: 集成处理逻辑到 UI，实现分析工作线程和参数调节
    status: completed
    dependencies:
      - implement-soft-mask
  - id: add-save-preview
    content: 实现音频保存和播放预览功能
    status: completed
    dependencies:
      - integrate-ui-logic
---

## 产品概述

创建一个名为 `harmony_remover.py` 的 PyQt6 应用，用于从人声干音中消除和音/合音，只保留目标歌手的主唱声音。

## 核心功能

- 加载 44100Hz WAV 格式的人声干音（包含主唱+和音混合）
- 波形可视化显示与频谱图
- 用户绘制参考区域标记"干净主唱"样本
- 基于频谱掩码算法提取目标人声
- 实时预览和播放控制
- 保存处理后的纯净人声

## 技术原理

1. 使用 STFT 将音频转换到频域
2. 从参考区域提取主唱的频谱特征（平均频谱指纹）
3. 计算每个时间帧与参考特征的余弦相似度
4. 基于相似度生成软掩码（soft mask），相似度高保留，相似度低抑制
5. 应用掩码后通过 ISTFT 重构音频

## 界面与操作逻辑

复用 `singer_cleaner.py` 的界面架构：

- 模型配置区域 → 简化为仅显示应用信息（无需加载外部模型）
- 音频加载、播放、停止控制
- 波形显示与交互（绘制/移动/调整参考区域）
- 参数调节滑块（掩码强度/灵敏度）
- 状态栏进度显示
- 保存处理结果

## 技术栈

- **GUI 框架**: PyQt6 + PyQt6-Qt6
- **音频处理**: numpy, scipy (STFT/ISTFT), soundfile
- **频谱分析**: librosa（可选，用于高级频谱特征）
- **音频播放**: ffplay (FFmpeg) 子进程

## 实现方案

### 核心算法流程

1. **STFT 变换**: 使用 `scipy.signal.stft` 将时域音频转换为时频表示
2. **参考特征提取**: 对选区音频计算平均频谱幅度，作为主唱音色指纹
3. **相似度计算**: 逐帧计算余弦相似度 `cosine_similarity = dot(ref, frame) / (norm(ref) * norm(frame))`
4. **软掩码生成**: `mask = sigmoid((similarity - threshold) * sensitivity)`，实现平滑过渡
5. **频域掩码**: `enhanced_spectrogram = original_spectrogram * mask[:, None]`
6. **ISTFT 重构**: 使用 `scipy.signal.istft` 转换回时域音频

### 架构设计

采用与 `singer_cleaner.py` 一致的分层架构：

- **UI 层**: 主窗口类 `HarmonyRemoverApp`，布局管理
- **可视化层**: `WaveformWidget` 复用/适配，显示波形+频谱+选区
- **处理层**: `HarmonyRemovalWorker` (QThread)，后台执行 STFT/掩码/ISTFT
- **音频 I/O**: soundfile 读写，ffplay 播放

### 关键设计决策

1. **无需外部模型**: 完全基于信号处理算法，无需加载 VAD 或 Speaker 识别模型，简化部署
2. **软掩码替代硬阈值**: 避免音频片段的突兀切换，保留自然过渡
3. **灵敏度参数可调**: 用户可控制和音抑制的激进程度
4. **MPS/GPU 无关**: 纯 CPU 的 numpy/scipy 运算已足够高效，无需 GPU 加速

## 目录结构

```
/Users/lpcw/Documents/PCS/audio-edit/
├── harmony_remover.py          # [NEW] 主应用文件，基于 singer_cleaner.py 架构
│   # 包含: HarmonyRemoverApp, WaveformWidget, HarmonyRemovalWorker
│   # 核心算法: STFT, 频谱特征提取, 软掩码生成, ISTFT
```

## 实现注意事项

1. **STFT 参数**: 窗口大小 2048，hop length 512，Hann 窗，保持相位一致
2. **参考区域处理**: 支持多段选区，合并计算平均频谱特征
3. **掩码平滑**: 在时间轴上应用轻微平滑，避免频谱闪烁
4. **内存管理**: 长音频分段处理，避免一次性加载全部频谱数据
5. **复用现有模式**: 严格遵循 `singer_cleaner.py` 的代码风格、信号命名、线程模式