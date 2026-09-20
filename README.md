# Si3N4 势函数训练成功框架

## 项目概述

氮化硅机器学习势函数完整训练框架，两阶段策略：

1. 无应力训练，获得高精度力/能量模型
2. 极低应力微调，引入应力预测

## 数据

* 5000 帧（4000 训练 / 500 验证 / 501 测试）
* 原子数 252（Si=108, N=144）
* 路径：data/Si3N4\_5000/

## 模型架构

* 描述符：se\_e2\_a
* 嵌入网络：\[25, 50, 100]
* 拟合网络：\[200, 200, 200]
* 截断半径：6.0 埃

## 两阶段训练

### 阶段一：无应力训练

* 配置：configs/no\_stress.json
* 步数：50000
* 损失权重：pref\_e 0.02 到 1.0，pref\_f 1000 到 1.0，pref\_v = 0
* 输出：models/graph\_no\_stress\_new.pb

### 阶段二：应力微调

* 配置：configs/stress\_finetune.json
* 步数：10000
* 损失权重：pref\_v 0.001 到 0.01
* 起点：graph\_no\_stress\_new.pb
* 输出：models/graph\_stress\_finetuned.pb

## 最终精度（测试集）

|指标|无应力模型|微调模型|
|-|-|-|
|Energy MAE/atom|0.287 meV|0.307 meV|
|Force MAE|0.00696 eV/A|0.00787 eV/A|
|Virial MAE/atom|未训练|0.144 eV|

## 目录结构

data/          数据集
descriptor/    描述符源码
configs/       训练配置
models/        冻结模型
checkpoints/   检查点
logs/          训练日志
scripts/       训练脚本

## LAMMPS 使用

pair\_style deepmd models/graph\_stress\_finetuned.pb
pair\_coeff \* \* Si N

