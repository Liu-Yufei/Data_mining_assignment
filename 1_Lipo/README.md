## 🧪 药物脂溶性预测（Lipophilicity Prediction）

本项目基于 **图神经网络（GNN）** 中的 **GINEConv** 模型，预测分子在 **pH 7.4** 条件下的辛醇/水分配系数（logD），属于 **回归任务**，目标是将预测的 RMSE 控制在 0.7 以下。

---

### 📂 数据集介绍

数据来源：[ChEMBL](https://www.ebi.ac.uk/chembl/) 数据库的 Lipophilicity 数据集。

---

### 🧠 模型结构

本项目使用 PyTorch Geometric 实现 GINE 网络。模型主要结构如下：

* 两层 `GINEConv` 图卷积层（使用边特征）
* 全局均值池化（`global_mean_pool`）
* 两层全连接层
* Dropout 正则化

---

### 🚀 使用方法

#### 1. 安装依赖

```bash
conda create -n lipo python=3.10
conda activate lipo
pip install -r requirements.txt
```

#### 2. 训练模型

```bash
python main.py --mode 0
```

* 训练集：`../Dataset/1_Lipophilicity/LIPO_train.csv`
* 模型将保存为：`1_LIPO_best_model_1.pt`

#### 3. 使用已训练模型评估测试集

```bash
python main.py --mode 1
```

* 会加载 `1_LIPO_best_model_1.pt` 并评估测试集 RMSE。

---

### ⚙️ 参数说明（命令行）

| 参数       | 类型    | 默认值 | 说明              |
| -------- | ----- | --- | --------------- |
| `--mode` | `int` | `1` | `0` 训练；`1` 加载测试 |

