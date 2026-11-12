# Pump it Up: Data Mining the Water Table

> DrivenData 练习赛 · 坦桑尼亚水泵运行状态三分类

- **赛事主页**：<https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/>
- **题型**：表格数据多分类（tabular multiclass classification）
- **难度**：intermediate practice（长期开放的练习赛，无截止压力）
- **数据来源**：[Taarifa](http://taarifa.org/)（开源基础设施众包报修平台）+ [坦桑尼亚水利部](http://maji.go.tz/)

---

## 1. 赛题背景

坦桑尼亚大量农村供水点（水井、手压泵、 communal standpipe 等）建成后缺乏维护跟踪，很多水泵在损坏后长期无人修复，直接影响当地居民能否获得干净、安全的饮用水。

Taarifa 把各地上报的水泵信息汇聚成一张「现实世界版的 bug tracker」：每个水井的位置、类型、安装方、资金来源、管理方式、水质水量等都有记录，并附带一次现场调查判定的运行状态。

**核心洞察**：如果能在水泵彻底坏掉之前，从这些静态属性中预测出哪些水井「即将出问题」或「已经坏了」，维护资源就可以优先投向高风险水井，而不是等坏了再去修。

## 2. 任务定义

给定每个水泵（waterpoint）的一组属性，预测其当前的运行状态 `status_group`，共三个类别：

| 标签 | 含义 |
| --- | --- |
| `functional` | 正常运行，无需维修 |
| `functional needs repair` | 仍在使用，但需要维修 |
| `non functional` | 已停止运行 |

即：**输入** 40 列左右的水泵属性（分类 + 数值 + 时间 + 地理坐标），**输出** 三选一的类别标签。

类别分布并不均衡（实测：`functional` 54.31%、`non functional` 38.42%、`functional needs repair` 仅 7.27%），因此这是一个**不均衡三分类**问题，`needs repair` 这一类是难点，也是拉开分数差距的地方。

## 3. 数据与字段

原始数据已就位，四个文件都在 `data/raw/`，并随仓库一起提交（`.gitattributes` 已关闭这几个文件的换行转换，保证检出后字节不变、MD5 可对上）：

| 文件 | 内容 | 形状 | 大小 | MD5 |
| --- | --- | --- | --- | --- |
| `TrainingSetValues.csv` | 训练集特征 | 59,400 × 40 | 20.1 MB | `7f81583401fcd17fd588208179f22441` |
| `TrainingSetLabels.csv` | 训练集标签（`id`, `status_group`） | 59,400 × 2 | 1.1 MB | `743edfcd5364347eae58377a8788d54d` |
| `TestSetValues.csv` | 测试集特征 | 14,850 × 40 | 5.0 MB | `76b3f3b6ae95a0fc283c8f7afab09017` |
| `SubmissionFormat.csv` | 提交模板 | 14,850 × 2 | 324 KB | `65a2b61efb55d701c64a36f75448d1ec` |

拿到数据后可先校验一遍，避免队友之间文件不一致：

```powershell
Get-ChildItem data\raw\*.csv | Get-FileHash -Algorithm MD5
```

训练集 / 测试集按 `id` 划分，两边的 `id` 集合完全不重叠（0–74,247 共 74,250 个水井），`date_recorded` 跨度 2002-10-14 至 2013-12-03，`construction_year` 取值 0–2013（0 表示未知），`gps_height` 为 −90 至 2,770。

### 3.1 字段一览

| 字段 | 说明 |
| --- | --- |
| `id` | 行唯一标识，提交时使用 |
| `amount_tsh` | 总静水头（可用水量），大量为 0，缺失语义明显 |
| `date_recorded` | 该行记录的日期 |
| `funder` | 出资方 |
| `gps_height` | 井所在海拔 |
| `installer` | 安装方 |
| `longitude` / `latitude` | GPS 坐标（存在 0 值与偏移异常） |
| `wpt_name` | 水井名称 |
| `num_private` | 私有化程度相关字段（文档留空） |
| `basin` | 所属流域 |
| `subvillage` | 行政村以下地名 |
| `region` / `region_code` | 地区及其编码 |
| `district_code` | 区县编码 |
| `lga` | 地方政府辖区 |
| `ward` | 选区 |
| `population` | 水井周边人口 |
| `public_meeting` | 是否开过公众会议（True/False，含缺失） |
| `recorded_by` | 记录该行的调查组 |
| `scheme_management` | 供水系统的管理方 |
| `scheme_name` | 供水系统名称 |
| `permit` | 是否持有取水许可 |
| `construction_year` | 建成年份（含 0 表示未知） |
| `extraction_type` / `_group` / `_class` | 取水方式，三列为由细到粗的层级 |
| `management` / `management_group` | 管理方式及大类 |
| `payment` / `payment_type` | 用水付费方式，两种粒度（58.2% 行取值相同） |
| `water_quality` / `quality_group` | 水质及大类（仅 13.0% 行字面相同，是两种粒度） |
| `quantity` / `quantity_group` | 水量及大类（两列 100% 相同，纯重复） |
| `source` / `source_type` / `source_class` | 水源，三列为由细到粗的层级 |
| `waterpoint_type` / `waterpoint_type_group` | 水井类型及大类 |

### 3.2 数据的几个坑（建模时优先处理）

实测数据画像（训练集 59,400 行）：

| 现象 | 实测情况 |
| --- | --- |
| 显式缺失 | `scheme_name` 28,810（48.5%）、`scheme_management` 3,878、`installer` 3,655、`funder` 3,637、`public_meeting` 3,334、`permit` 3,056、`subvillage` 371、`wpt_name` 2 |
| 伪缺失（0 值） | `amount_tsh` 70.10%、`population` 35.99%、`construction_year` 34.86%、`gps_height` 34.41%、`longitude` 3.05%、`latitude` 0% |
| 常量列 | `recorded_by` 是唯一整列常量的字段（训练/测试都只有 `GeoData Consultants Ltd`），零信息量，直接删；`num_private` 98.7% 为 0（65 个取值），近似常量 |
| 完全重复列 | `quantity` 与 `quantity_group` 逐行 100% 相同，删其一即可 |
| 高基数类别 | `wpt_name` 37,399、`subvillage` 19,287、`scheme_name` 2,695、`installer` 2,145、`funder` 1,896、`ward` 2,092、`lga` 125、`region` 21、`basin` 9 |
| 地理编码不自洽 | `region` 只有 21 个取值，`region_code` 却有 27 个；`district_code` 只有 20 个取值，与该层级应有的数量对不上 —— 这些 code 列不能当作 `region`/`lga` 的干净替代品 |
| 退化坐标 | `latitude` 最大值约 −2e−08、`longitude` 最小值 0.0，即存在落在 (0, 0) 附近「零点」的记录，需要当作缺失剔除 |
| 布尔列带空洞 | `public_meeting` / `permit` 只有 True/False 两个取值，但分别有 5.61% / 5.14% 缺失，读进来是 object，需显式转 `boolean` 或三态编码 |

对应的处理要点：

1. **零信息列先删**：`recorded_by` 全列常量、`quantity_group` 与 `quantity` 完全重复，都该在第一轮就剔掉，否则会稀释特征重要性。
2. **伪缺失**：`construction_year = 0`、`gps_height = 0`、`longitude/latitude = 0`、`population = 0` 都代表「没记录」而非真实取值，需显式转成缺失再插补，并保留 `is_missing` 指示变量（缺失本身可能就是信号，例如 `scheme_name` 缺失近一半）。
3. **层级重复列**：`extraction_type*`、`source*`、`waterpoint_type*`、`management*`、`payment*` 都由细到粗多份编码，直接 one-hot 会造成共线与冗余，建议每组只保留一列或用层级结构派生。
4. **高基数类别**：`wpt_name` 近 3.7 万个取值、`subvillage` 近 2 万个，多数只出现一次，需要做频次截断（如只留 top-N + `other`）、目标编码（必须放在交叉验证内部做）或干脆只抽取后缀/长度这类结构化特征。
5. **脏字符串**：`funder` / `installer` 里有大量同义异写（大小写、缩写、拼写错误、机构改名），清洗归一化后这类字段的可用性会明显提升。
6. **地理强信号**：经纬度与 `region`/`lga` 高度相关，直接用编码会丢掉空间连续性；聚类、到最近城市的距离、邻域内目标均值等空间特征通常更有效。

### 3.3 实测的强信号（建模前先知道事半功倍）

单看某个类别字段的目标分布，已经能看出哪些特征值得重点做：

| 字段 | 取值 | 样本数 | functional | needs repair | non functional |
| --- | --- | --- | --- | --- | --- |
| `quantity` | `dry` | 6,246 | 2.5% | 0.6% | **96.9%** |
| `quantity` | `enough` | 33,186 | 65.2% | 7.2% | 27.5% |
| `waterpoint_type` | `other` | 6,380 | 13.2% | 4.6% | **82.2%** |
| `extraction_type_class` | `other` | 6,430 | 16.0% | 3.2% | **80.8%** |
| `payment_type` | `never pay` | 25,348 | 44.9% | 7.5% | **47.6%** |
| `payment_type` | `annually` | 3,642 | 75.2% | 6.8% | 18.0% |
| `source_class` | `surface` | 13,328 | 54.5% | **11.8%** | 33.7% |
| `source_class` | `groundwater` | 45,794 | 54.2% | 6.0% | 39.8% |

两个直接结论：

- `quantity = dry` 一个字段就几乎锁定了 `non functional`（96.9%），而 `waterpoint_type / extraction_type_class = other` 也是同类强信号 —— 这类「明显坏了」的样本模型很容易学。
- 真正难的是 **`needs repair`**：它整体只占 7.27%，且没有任何单字段能把它分离出来。唯一明显的差异是 `source_class = surface` 的 needs-repair 率（11.8%）约为 groundwater（6.0%）的两倍。要提升这一类的召回，基本只能靠多特征交互 + 类权重/重采样，而不是找某个「银弹字段」。

## 4. 评估与提交

- **评估指标**：分类准确率（categorization accuracy），即 `status_group` 预测正确的比例。
- **排行榜**：分公开榜与私有榜，最终排名以私有榜为准。
- **提交文件**：两列 CSV（`id`, `status_group`），行数须与 `SubmissionFormat.csv` 一致（14,850 行）。

```csv
id,status_group
50785,functional
51630,functional
17168,functional
45559,functional
49871,functional
```

参考基线：全部预测 `functional` 的准确率是 **0.5431**（等于该类的样本占比），这是任何模型必须越过的下限；常见的强 baseline（LightGBM + 基础特征工程）大致在 0.80 上下，头部成绩约 0.82–0.83。

## 5. 本项目要实现的目标

### 5.1 总目标

在本地构建一条**可复现**的完整建模流水线：从原始 CSV 出发，产出符合格式的 `submission.csv`，并在 DrivenData 榜单上取得**不低于 0.82 的准确率**（以私有榜为准，当前集成 CV 0.8147，公开榜榜首 0.8325），同时把特征处理与模型选择的过程记录成可读的实验笔记。

### 5.2 分阶段目标

| 阶段 | 要做的事 | 完成标准 | 状态 |
| --- | --- | --- | --- |
| 0. 数据获取 | 注册赛事、下载四个 CSV、核对行列数与字段 | 数据落在 `data/raw/`，形状与字段清单记录在案 | ✅ 已完成 |
| 1. EDA | 标签分布、缺失模式、类别基数、地理分布、与目标的关系 | 产出 EDA 笔记，列出至少 5 条可执行的特征假设 | ✅ 已完成 |
| 2. 数据清洗 | 伪缺失修正、`funder`/`installer` 归一化、层级列去冗余、日期拆分 | 清洗函数可复用，train/test 处理逻辑完全一致 | ✅ 已完成 |
| 3. 特征工程 | 频次编码、目标编码（交叉验证内做）、地理聚类/近邻特征、井龄、文本型名称的简化特征 | 特征表落盘，含特征重要性排序 | 🚧 进行中（TE 已测，geo/文本实测中性） |
| 4. 基线模型 | 逻辑回归 / 随机森林 快速跑通全流程 | 生成第一份可提交的 `submission.csv` | ✅ 已完成 |
| 5. 主模型 | LightGBM / XGBoost / CatBoost，处理类别不均衡 | 本地 CV 准确率与公开榜分数接近，无明显过拟合 | ✅ 已完成（CatBoost 0.8115） |
| 6. 集成与调参 | 多模型 soft-voting / stacking，超参搜索 | 公开榜 ≥ 0.81 | ✅ 已完成（集成 CV 0.8147） |
| 7. 沉淀 | 复现说明、实验记录、错误分析（哪类水井最易判错） | README 与实验笔记完整，他人可一键复跑 | ⏳ 待开始 |

> 阶段 1–7 的具体做法、已测基线与优先级排序见 **[docs/技术方案.md](docs/技术方案.md)**。
> 阶段 1 的 EDA 结果见 `notebooks/01_eda.ipynb`（本地，不提交），每次实验自动追加到 `docs/experiments.csv`（本地，不提交）。

### 5.3 明确的非目标

- 不追求刷到排行榜第一（练习赛，收益递减）。
- 不引入外部数据集（保持与赛题口径一致）。
- 不做在线服务化 / 部署，只做离线训练与提交。

## 6. 环境与复现

依赖用 [uv](https://docs.astral.sh/uv/) 管理，环境定义全部落在三个文件里，队友照着做即可得到逐字节一致的依赖树：

| 文件 | 作用 |
| --- | --- |
| `pyproject.toml` | 声明运行时依赖与 `dev` 依赖组（jupyterlab / ipykernel / ruff） |
| `uv.lock` | 锁文件，记录 117 个包的精确版本 + 传递依赖 + 哈希，**复现以它为准** |
| `.python-version` | 固定 CPython 3.12，uv 会据此自动挑选/下载解释器 |

### 6.1 队友复现步骤

先装 uv（Windows）：

```powershell
winget install --id=astral-sh.uv -e
# 或者：powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

然后在仓库目录：

```powershell
git clone <repo> && cd Pump-it-Up
uv sync --locked          # 按 uv.lock 建 .venv 并安装依赖，不会重新解析版本
uv run python -c "import pandas, sklearn, lightgbm; print('ok')"
```

- `uv sync --locked` 里的 `--locked` 是关键：如果 `pyproject.toml` 和 `uv.lock` 不一致，它会直接报错而不是悄悄升级到新版本。
- 跑脚本用 `uv run python src/xxx.py`，不需要手动 activate。
- 跑 notebook 用 `uv run jupyter lab`，内核选 venv 自带的那个即可（`uv sync` 已装好 ipykernel）。本机额外注册过别名 `Python (Pump-it-Up)`，那只是本地机器上的便利，队友不需要。
- 想脱离 uv 用别的工具，可 `uv export --format requirements-txt --no-hashes -o requirements.txt` 导出（导出的文件请勿提交，避免和 `uv.lock` 漂移）。

### 6.2 当前锁定版本

| 包 | 版本 | 包 | 版本 |
| --- | --- | --- | --- |
| Python | 3.12.13 | LightGBM | 4.7.0 |
| NumPy | 2.5.4 | XGBoost | 3.4.1 |
| pandas | 3.1.1 | CatBoost | 1.2.10 |
| SciPy | 1.18.1 | Matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 | seaborn | 0.13.2 |

> 注：本机系统自带的 Python 是 MinGW/ucrt64 构建的 3.11.9，pandas、scikit-learn、LightGBM 这些只提供 MSVC wheel 的包在它上面装不上，所以务必走上面的 uv 流程，不要用系统 `python`。

### 6.3 新增依赖

```powershell
uv add polars            # 加到 [project].dependencies
uv add --dev pytest      # 加到 dev 组
uv lock && uv sync       # 更新锁文件并同步环境
```

改完依赖记得把 `pyproject.toml` 和 `uv.lock` 一起提交，否则队友复现出来的环境会不一致。

## 7. 目录结构

```text
Pump-it-Up/
├── README.md              # 本文件：赛题说明与目标
├── pyproject.toml         # 依赖声明（uv）
├── uv.lock                # 锁文件，复现依据
├── .python-version        # 固定 Python 3.12
├── .gitignore
├── .gitattributes         # 数据文件禁止换行转换，保证 MD5 可复现
├── .venv/                 # uv sync 生成，不入库
├── data/
│   ├── raw/               # 官方四个 CSV，已入库（MD5 见 §3）
│   └── processed/         # folds.npy、oof_*.npy，不入库
├── notebooks/             # 本地分析（不提交，也不忽略）
│   └── 01_eda.ipynb       # 阶段 1：EDA 与基线诊断（含图与输出）
├── src/
│   ├── config.py          # 路径、类别、随机种子等常量
│   ├── data.py            # 读取、轻量清洗、固定 fold 划分
│   ├── features.py        # 特征构造（E1 基线特征集）
│   ├── train.py           # 交叉验证、OOF、实验登记
│   └── evaluate.py        # 指标、混淆矩阵、提交校验
├── submissions/           # 各版本提交文件，不入库
└── docs/
    ├── 技术方案.md         # 技术路线与待验证假设
    ├── experiments.csv    # 实验登记（本地，不提交）
    └── figures/           # EDA 图（不入库）
```

## 8. 参考

- [赛事主页](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)
- [Problem Description（字段与标签定义）](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/)
- [About Taarifa（数据背景）](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/24/)
- [Taarifa 项目](http://taarifa.org/) · [坦桑尼亚水利部](http://maji.go.tz/)
