import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
from tqdm import tqdm
import os

IN_DIR = Path("./4data")
OUT_DIR = Path("./datas")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------- I/O 工具 -------------------------------

def looks_like_sample_id(s: str) -> bool:
    """根据你数据常见模式判断是否像样本ID（CRA***, CRB***, HP*** 等）"""
    if not isinstance(s, str):  # 字符串类型检查
        return False
    return bool(re.match(r"^(CR[A-Z]\d+|HP\d+|[A-Z]{2,4}\d{2,4}[A-Za-z]?)$", s))  # 正则匹配

def read_table_robust(path: Path) -> pd.DataFrame:
    """
    解决生物信息学数据中常见的文件后缀与实际格式不符（例如：明明是文本文件却挂着 .xls 后缀），尽可能将其读成表格形式
    健壮读取：
    1) 先按 Excel 引擎读（.xlsx/.xls）。若只读出1列，视为“伪装Excel”的文本，转入CSV分支；
    2) 依次尝试 \t、,、; 分隔；
    3) 兜底用空白分隔。
    """
    path = Path(path)

    # 1) 真 Excel 优先
    if path.suffix.lower() in {".xlsx", ".xls"}:
        for engine in ("openpyxl", "xlrd"):  # 用专业的 Excel 引擎去读取
            try:
                df = pd.read_excel(path, engine=engine)
                if df.shape[1] >= 2:  # 读出来的列数≥2，直接返回数据
                    return df
            except Exception:
                pass  # 继续尝试文本读取

    # 2) 文本分隔优先尝试 TSV（你的 .xls 多为 TSV）
    # 依次尝试常见分隔符
    for sep in ("\t", ",", ";"):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    # 3) 空白分隔兜底
    try:
        df = pd.read_csv(path, delim_whitespace=True, engine="python")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    raise ValueError(f"无法正确解析表格：{path}（可能是特殊分隔或编码）")

def _sanity(name: str, df: pd.DataFrame):
    """读完立即做健壮性检查，防止“只有一列”的静默错误。"""
    if df.shape[1] <= 1:
        raise ValueError(f"[{name}] 只读到 {df.shape[1]} 列，请检查分隔符/文件格式。")

def find_input_file(default_patterns: List[str], override: Optional[str]) -> Path:
    """
    优先读取用户手动指定的文件路径，如果用户没指定，就自动去默认文件夹（./4data）里按预设的文件名规则把需要的数据找出来
    严格大小写匹配：只按传入的 pattern 在 ./4data 下找；
    如果提供了 --ko / --species / --label 则直接用该路径。
    """
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"指定文件不存在：{p}")
        return p
    for pat in default_patterns:  # 区分大小写
        hits = sorted(IN_DIR.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"在 {IN_DIR} 未找到：{default_patterns}")

# ------------------------------- 表格规范化 -------------------------------

def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一成：行=样本(sample_id)，列=特征
    规则：
    - 若第一列像 sample_id / sample / id，则直接使用
    - 若第一列像 feature 名（Genus/K/KO/Species/OTU/ASV 等）或多数列头像样本ID，则视为“特征×样本”表 -> 转置
    - 其他情况启发式判断
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]  # 去重未命名列
    first_col = df.columns[0].lower()  # 提取第一列的列名，并变成小写（存为 first_col）

    if first_col in {"sample_id", "sample", "id"}:  # 已经是标准的行是样本
        out = df.rename(columns={df.columns[0]: "sample_id"})  # 重命名标准的sample_id
    else:  # 转置
        sample_like_cols = [c for c in df.columns[1:] if looks_like_sample_id(str(c))]
        if first_col in {"genus", "species", "otu", "asv", "k", "ko", "feature"} or \
           (len(sample_like_cols) >= max(3, int(0.5 * (len(df.columns) - 1)))):  # 如果第一列是特征
            feat_col = df.columns[0]
            df = df.set_index(feat_col).T
            df.index.name = "sample_id"
            out = df.reset_index()
        elif df.shape[0] > df.shape[1]:
            out = df.rename(columns={df.columns[0]: "sample_id"})
        else:
            out = df.T.reset_index().rename(columns={"index": "sample_id"})

    # 清理 sample_id 的空白与奇怪字符
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    # 去掉空 sample_id 的行
    out = out[out["sample_id"] != ""]
    # 如有重复样本，保留首次出现
    if out["sample_id"].duplicated().any():
        out = out.drop_duplicates(subset=["sample_id"], keep="first")
    return out

def coerce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """将特征列转为数值；保留 sample_id/label 原样。"""
    df = df.copy()
    non_feat = {"sample_id", "label"}
    for c in df.columns:
        if c not in non_feat:  # 不是样本和标签的，就是特征
            df[c] = pd.to_numeric(df[c], errors="coerce")  # 强制转换为数值类型
    return df

def align_by_intersection(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """基于样本ID交集对齐，并按 sample_id 排序，保证每个表的样本ID顺序完全一致。"""
    a_ids = set(a["sample_id"].astype(str))  # 取出表a的sample_id列并转为字符串格式
    b_ids = set(b["sample_id"].astype(str))
    common = sorted(a_ids & b_ids)  # 求交集并排序
    a2 = a[a["sample_id"].astype(str).isin(common)].copy().sort_values("sample_id")
    b2 = b[b["sample_id"].astype(str).isin(common)].copy().sort_values("sample_id")
    return a2, b2, common

# ------------------------------- 标签处理 -------------------------------

def normalize_label_values(s):
    """将多种标签写法归一到 {0,1}；无法识别的则返回 None。"""
    if pd.isna(s):
        return None
    x = str(s).strip().lower()
    if x in {"1", "case", "patient", "pos", "positive", "disease"}:  # 阳性/患病组记为1
        return 1
    if x in {"0", "control", "neg", "negative", "healthy"}:  # # 阴性/健康组记为0
        return 0
    try:
        return int(float(x))
    except Exception:
        return None

def load_label_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    自动找到标签文件，从中把代表“样本名”和“临床分组（如患病/健康）”的两列数据揪出来，统一改名，并做最终的清洗
    读取 label 表（若提供），并尝试标准化列名。
    自动发现严格大小写：Sample.Info.*
    """
    if path is None:
        candidates = sorted(list(IN_DIR.glob("Sample.Info.*")))  # 注意大小写
        if not candidates:
            return None
        path = candidates[0]

    df = read_table_robust(path)
    # 标准化列名（不改大小写存储，只在逻辑上识别）
    cols_lower = {c.lower(): c for c in df.columns}
    # 识别并统一"样本名"列名字为"sample_id"
    if "sample_id" not in cols_lower:
        for cand in ("sample", "id", "sampleid"):
            if cand in cols_lower:
                df = df.rename(columns={cols_lower[cand]: "sample_id"})
                break
    else:
        df = df.rename(columns={cols_lower["sample_id"]: "sample_id"})
    # 识别并统一"分组/标签"列名字为"label"
    if "label" not in cols_lower:
        for cand in ("y", "group", "status", "case_control"):
            if cand in cols_lower:
                df = df.rename(columns={cols_lower[cand]: "label"})
                break
    else:
        df = df.rename(columns={cols_lower["label"]: "label"})

    if "sample_id" not in df.columns or "label" not in df.columns:
        return None

    # 提取与清洗
    df = df[["sample_id", "label"]].copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["label"] = df["label"].map(normalize_label_values)
    df = df.dropna(subset=["sample_id"])

    return df

def write_final_csv(feat_df: pd.DataFrame, label_df: Optional[pd.DataFrame], common_ids: List[str], out_path: Path):
    """
    将 label 合并到对齐后的特征表：
    - 若无 label_df，则输出 *_features_no_label.csv
    - 若有 label_df，则内连接并剔除缺失 label，label 放在第2列
    """
    if label_df is None:
        out_path = out_path.with_name(out_path.stem.replace("_abundance", "") + "_features_no_label.csv")
        feat_df.to_csv(out_path, index=False)
        return None, out_path

    ldf = label_df[label_df["sample_id"].astype(str).isin(common_ids)].dropna(subset=["label"])
    merged = feat_df.merge(ldf, on="sample_id", how="left")  # 合并特征表与标签表
    merged = merged.dropna(subset=["label"]).copy()

    # label 移到第2列
    cols = list(merged.columns)
    cols.remove("label")
    cols.insert(1, "label")
    merged = merged[cols]

    merged.to_csv(out_path, index=False)  # 将最终的数据表存为 CSV 文件
    return merged, out_path

# ------------------------------- 主流程 -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ko", help="KO 表文件路径（默认自动匹配 ./4data/KO.profile.*）")
    parser.add_argument("--species", help="species 表文件路径（默认自动匹配 ./4data/species.profile.*）")
    parser.add_argument("--label", help="label 表文件路径（默认自动匹配 ./4data/Sample.Info.*）")
    args = parser.parse_args()

    # 设置进度条
    steps = [
        "查找输入文件",
        "读取 KO 表",
        "读取 species 表",
        "准备 KO 表",
        "准备 species 表",
        "对齐样本交集",
        "读取 label",
        "写出最终 CSV"
    ]

    pbar = tqdm(total=len(steps), desc="处理进度")

    # 1 查找输入文件的路径
    ko_path = find_input_file(["KO.profile.*"], args.ko)
    sp_path = find_input_file(["species.profile.*"], args.species)
    label_path = Path(args.label) if args.label else None
    pbar.update(1)

    # 2-3 读取
    ko_raw = read_table_robust(ko_path); _sanity("KO", ko_raw); pbar.update(1)
    sp_raw = read_table_robust(sp_path); _sanity("species", sp_raw); pbar.update(1)

    # 4-5 统一形态、转置、数值化
    ko_prep = coerce_numeric_features(prepare_table(ko_raw))
    ko_prep = ko_prep.rename(columns={ko_prep.columns[0]: "sample_id"}); pbar.update(1)
    sp_prep = coerce_numeric_features(prepare_table(sp_raw))
    sp_prep = sp_prep.rename(columns={sp_prep.columns[0]: "sample_id"}); pbar.update(1)

    print(f"[CHECK] KO: 样本数={ko_prep.shape[0]}, 特征数={ko_prep.shape[1]-1}, 示例ID={ko_prep['sample_id'].head(3).tolist()}")
    print(f"[CHECK] species: 样本数={sp_prep.shape[0]}, 特征数={sp_prep.shape[1]-1}, 示例ID={sp_prep['sample_id'].head(3).tolist()}")

    # 6 对齐交集
    ko_aln, sp_aln, common_ids = align_by_intersection(ko_prep, sp_prep)
    pd.DataFrame({"sample_id": common_ids}).to_csv(OUT_DIR / "samples_intersection.csv", index=False)
    pbar.update(1)

    # 7 读取 label
    label_df = load_label_table(label_path)
    pbar.update(1)

    # 8 合并并写出
    ko_final, ko_out = write_final_csv(ko_aln, label_df, common_ids, OUT_DIR / "ko_abundance.csv")
    sp_final, sp_out = write_final_csv(sp_aln, label_df, common_ids, OUT_DIR / "species_abundance.csv")
    pbar.update(1)

    pbar.close()
    print("[完成] 数据已处理")

if __name__ == "__main__":
    main()
