## 🧪 药物副作用预测（Lipophilicity Prediction）

本项目基于 [SIDER](http://sideeffects.embl.de/) 药物不良反应数据库，使用图神经网络（GCN）对化合物的 SMILES 表达式进行图结构建模，以预测其可能的副作用（多标签分类任务）。
```
2_sider/ # 药物副作用预测
│
├── Sider.py # 运行代码
├── 2_sider.docx # 报告
├── best_model.pth # 权重文件
├── curves_micro.png # AUROC和AUPR文件
├── README.md # 项目说明文档
└── requirements.txt # 环境要求文档
```

---

### 📂 数据集介绍

SIDER 是一个药物副作用数据库，每一行包含：

* 第一列：药物的 SMILES 字符串（分子结构的字符串表示）
* 后续列：每个副作用标签（0 表示无副作用，1 表示存在副作用）

### 数据格式示例：

```
SMILES,Effect1,Effect2,...,EffectN
CC(=O)OC1=CC=CC=C1C(=O)O,0,1,...,0
...

```

---

### 🧠 模型结构

* 基于 `GCNConv` 构建的深层图神经网络
* 节点特征包括：原子序数、是否芳香、是否在环中等 15 项
* 边特征包括：键类型、是否在环中等 3 项
* 采用三种图池化方式（mean/max/add）进行全图表征
* 使用三层全连接网络进行分类

---

### 🚀 使用方法

#### 1. 安装依赖

```bash
conda create -n sider python=3.10
conda activate sider
pip install -r requirements.txt
```

#### 2. 训练模型

```bash
python Sider.py --mode 0
```

* 默认最多训练 1000 轮
* 训练过程中会保存 `AUROC > 0.7` 的最优模型到 `best_model_1.pth`
* 最终生成 `ROC/PR` 曲线图 `curves_micro.png`

#### 3. 使用已训练模型评估测试集

```bash
python Sider.py --mode 1
```

* 加载 `best_model.pth` 并在测试集上进行评估
* 输出 AUROC 和 AUPR，并绘制曲线图
  
---

### ⚙️ 参数说明（命令行）

| 参数       | 类型    | 默认值 | 说明              |
| -------- | ----- | --- | --------------- |
| `--mode` | `int` | `1` | `0` 训练；`1` 加载测试 |
