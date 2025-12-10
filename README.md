# U4EIP：基于 U-Net 的电磁特性恢复系统

一个基于卷积神经网络的电磁特性恢复系统，通过轻量级U-Net模型从模拟电磁波传播数据中精确重建介质介电常数分布。

---

## 设计原理

### 1. 问题定义
- 输入：模拟电磁波传播数据（64×64矩阵）
- 输出：地下介质介电常数分布（64×64矩阵）
- 任务：从噪声观测中检测并重建圆环特征

### 2. 系统架构

#### 数据层：输入信号生成（这里以圆环型输入信号为例）
圆环掩膜模式:
- 圆环区域：数值 = 0.9（代表高介电常数）
- 背景区域：0.05-0.15（代表低介电常数）
- 高斯噪声（σ=0.03）模拟测量误差
- 随机旋转用于数据增强

数据增强策略:
- 旋转增强：0度、90度、180度、270度旋转
- 噪声注入：高斯噪声σ=0.03增加真实感
- 归一化处理：逐样本最小-最大归一化保证稳定性

#### 模型层：轻量级U-Net（深度=3）
编码器（3层）:
  Conv2D(32) -> Conv2D(32) -> MaxPool
  Conv2D(64) -> Conv2D(64) -> MaxPool
  Conv2D(128) -> Conv2D(128) -> MaxPool

瓶颈层:
  Conv2D(256) x 2

解码器（3层，含跳接）:
  UpSample -> Concat -> Conv2D(128) x 2
  UpSample -> Concat -> Conv2D(64) x 2
  UpSample -> Concat -> Conv2D(32) x 2

输出层: Conv2D(1, activation='sigmoid')

架构优势:
- 参数数量：约0.5百万（轻量级，快速训练）
- 跳接连接保留空间特征
- Sigmoid输出在[0, 1]范围内用于二分类

#### 训练层：二值交叉熵优化
损失函数: 二值交叉熵
优化器: Adam（学习率=0.001）
正则化: 早停机制（耐心=10）
评价指标: MSE、MAE

#### 推理层：基于Sigmoid的后处理
增强函数:
$f(x) = 1 / (1 + exp(-scale × (x - threshold)))$

参数说明:
- threshold：决策边界（默认0.5）
- scale：陡峭程度（更大值 = 更陡峭的分离）
- 效果：从软决策平滑过渡到硬决策



---

### 3. 可配置架构
```
CONFIG = {
    'num_samples': 500,        # 数据集大小
    'image_size': (64, 64),    # 输入/输出分辨率
    'base_filters': 32,        # 基础滤波器数量
    'depth': 3,                # U-Net深度（3个编码器层）
    'epochs': 60,              # 最大训练轮数
    'batch_size': 32,          # 批量大小
    'learning_rate': 0.001,    # Adam学习率
}
```

### 4. 模块化代码结构
数据生成模块 -> 模型构建 -> 训练循环 -> 后处理和可视化
  (create_ring_mask)  (build_unet)  (model.fit)  (postprocess_prediction)
  (generate_synthetic)                            (visualize results)



---

## pipline

数据生成步骤:
  生成500个圆环样本(64×64)
  数据增强：随机旋转
  归一化处理：逐样本缩放

模型训练步骤:
  初始化轻量级U-Net
  二值交叉熵损失
  早停机制（耐心=10）
  恢复最优权重

性能评估步骤:
  计算验证集MSE和MAE
  绘制训练/验证损失曲线
  保存指标到metrics.txt

预测和增强步骤:
  模型推理测试样本
  基于Sigmoid的后处理
  生成对比可视化

输出生成步骤:
  training_loss.png（收敛曲线）
  prediction_comparison.png（4面板图）
  metrics.txt（性能摘要）

---

## 快速开始指南

### 安装环境
```
pip install tensorflow numpy matplotlib
```

### 运行训练
选项1：在Jupyter/IPython中运行
cd src && jupyter notebook main.ipynb

选项2：Python执行
cd src && python -c "exec(open('main.ipynb').read())"

### 查看结果
```
ls -la results/
输出：
training_loss.png          - 损失曲线收敛
prediction_comparison.png  - 4面板对比（输入、真值、原始、增强）
metrics.txt               - 性能摘要
```

---

## 预期结果

### 验证指标
- MSE：小于0.01（目标）
- MAE：小于0.05（目标）
- 收敛：通常30-50个轮次（含早停）

### 可视化组件
1. 输入（含噪声圆环）：噪声观测（0.9圆环+0.05-0.15背景）
2. 真标签：干净圆环掩膜（基准真值）
3. 原始预测：Sigmoid连续输出[0, 1]
4. 增强预测：后处理二值输出

---

## 物理解释

数值范围对应的物理含义:
0.9：高介电常数（圆环）- 材料例如：粘土、水、混凝土
0.05-0.15：低介电常数（背景）- 材料例如：空气、沙土、土壤

应用领域:
- 地面穿透雷达(GPR)：地下异常检测
- 医学成像：MRI/超声伪影移除
- 无损检测：工业缺陷检测
- 地球物理勘测：层界面识别

---

***Last Updated: 2025.12.11***  

