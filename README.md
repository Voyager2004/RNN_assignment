# 古诗生成模型优化版（基于 RNN）

本项目基于开源项目 [taishan1994/pytorch\_peot\_rnn](https://github.com/taishan1994/pytorch_peot_rnn) 进行修复与改进，旨在提升古诗生成效果，纠正原始代码中的数据划分和模型封装问题，并对超参数进行了重新设计优化。

---

## 🔧 更新与修复说明

**1. 修复训练集与测试集重叠的问题**

* 原始 `split_train_test` 函数中，训练集和测试集使用了相同数据
* 现已修正为：

  * 训练集：`data[:train_total]`
  * 测试集：`data[train_total:]`

**2. 修正生成函数中使用全局变量的问题**

* `Trainer.generate` 和 `Trainer.gen_acrostic` 中原本直接调用全局变量 `model`
* 现已改为 `self.model`，符合面向对象设计原则，避免状态混乱

**3. 修复函数调用参数不匹配问题**

* 原始脚本中 `get_data('./data/peot.txt', config)` 与函数定义不符
* 已修正为 `get_data(config)`

---

## ⚙️ 模型参数配置（调参后）

```python
self.num_epoch = 20
self.batch_size = 128
self.lr = 1e-3
self.weight_decay = 1e-4
self.max_gen_len = 200
self.max_len = 125
self.embedding_dim = 1000
self.hidden_dim = 512
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install torch numpy tqdm
```

### 训练与生成

```bash
python main.py
```

训练完成后会自动保存模型，并支持自由生成藏头诗或七言诗。

---

## 📂 数据说明

由于版权或体积问题，**本项目未附带训练语料文件 `peot.txt`**。
请手动前往原始项目获取语料文件，并放入如下路径：

```
pytorch_peot_rnn/
└── data/
    └── peot.txt   # 古诗语料文本（UTF-8 编码）
```

原项目地址：[https://github.com/taishan1994/pytorch\_peot\_rnn](https://github.com/taishan1994/pytorch_peot_rnn)

---

## 📄 文件结构

```
pytorch_peot_rnn/
├── checkpoints/         # 模型保存路径
├── data/                # 数据文件夹（需自行添加 peot.txt）
├── model.py             # RNN 模型定义
├── main.py              # 主运行脚本
├── process.py           # 数据预处理与序列构建
├── train_eval.py        # 训练与验证逻辑
├── utils.py             # 工具函数
├── README.md
```

---

## 📌 项目目标与意义

本项目用于探索中文自然语言生成的基本方法，适合用于中文文本建模、古典文学 AI 创作、RNN 训练调优等方向的入门实践。优化后的版本提高了训练稳定性与生成质量，便于教学与项目延伸使用。

---

## 🙋‍♀️ 联系与贡献

欢迎提交 issue、提出建议或 fork 后参与改进。
你可以在本项目基础上拓展支持：

* LSTM、GRU 替代基本 RNN
* Transformer-based 古诗生成
* 加入平仄对仗约束

