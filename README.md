# 信贷违约预测项目

本仓库用于保存百度实习项目“信贷违约预测”的代码、实验报告和阶段性分析结果。

项目目标是基于借款人的贷款信息、信用信息和历史行为特征，预测用户是否可能发生违约，为后续信贷风控建模和风险识别提供数据基础。

## 数据说明

完整数据文件体积较大，不上传到 GitHub，只保存在本地。

本地数据文件包括：

```text
credit_data.csv
credit_data_cleaned.csv
```

上述数据文件已通过 `.gitignore` 忽略，不纳入普通 Git 版本管理。

## 当前进度

目前已完成：

- 第一阶段：项目启动与初步数据探索
- 第二阶段：数据清洗与探索性数据分析

第二阶段完成了重复值检查、缺失值处理、业务逻辑异常检测、统计极端值处理、字符串和日期一致性检查，以及目标变量、数值字段、类别字段、匿名字段和时间趋势的探索性数据分析。

## 主要文件

```text
data_preprocessing.ipynb
data_preprocessing.py
第一阶段实验报告.docx
第一阶段实验报告.pdf
第二阶段实验报告.docx
第二阶段实验报告.pdf
```

文件说明：

- `data_preprocessing.ipynb`：记录第一阶段数据概览、第二阶段数据清洗和 EDA 的完整分析过程。
- `data_preprocessing.py`：正式代码交付文件，可从原始数据生成清洗后的数据。
- `第一阶段实验报告.docx` / `第一阶段实验报告.pdf`：第一阶段实验报告。
- `第二阶段实验报告.docx` / `第二阶段实验报告.pdf`：第二阶段实验报告。

## 运行方式

在项目目录下放置原始数据文件 `credit_data.csv` 后，可以运行：

```bash
python data_preprocessing.py
```

脚本默认读取：

```text
credit_data.csv
```

并输出：

```text
credit_data_cleaned.csv
```

也可以指定输入和输出路径：

```bash
python data_preprocessing.py --input credit_data.csv --output credit_data_cleaned.csv
```

## 下一阶段

第三阶段将重点进行特征工程和建模前数据准备，包括字段含义确认、数据集划分、日期特征构造、工作年限处理、类别字段编码、高基数字段处理和匿名字段处理策略设计。
