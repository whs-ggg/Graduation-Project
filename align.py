# -*- coding: utf-8 -*-
"""
对齐四模态 CSV 的样本顺序（以 KO 为基准），并直接覆盖原文件
会为每个原文件先生成 *.bak 备份
"""

import os
import pandas as pd

# ===== 修改这里 =====
DATA_DIR = "Data/CRC4"
FILES = [
    "ko_abundance.csv",            # 基准表（建议 KO）
    "species_abundance.csv",
    "untarget_pos_abundance.csv",
    "untarget_neg_abundance.csv",
]
# ====================

def read_clean(path):
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"{path} 缺少 sample_id 列")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    return df

def main():
    paths = [os.path.join(DATA_DIR, f) for f in FILES]
    dfs = [read_clean(p) for p in paths]

    # 基准 sample_id 顺序
    base = dfs[0]["sample_id"].tolist()
    base_set = set(base)
    print(f"[基准] {FILES[0]}: {len(base)} 样本")

    # 检查其它表集合
    for i in range(1, len(dfs)):
        name = FILES[i]
        df = dfs[i]
        cur_set = set(df["sample_id"].tolist())
        missing = base_set - cur_set
        extra   = cur_set - base_set
        if missing or extra:
            print(f"[错误] {name} 样本集合不一致：")
            if missing:
                print(f"  - 缺少 {len(missing)} 个: {list(sorted(missing))[:10]} ...")
            if extra:
                print(f"  - 多出 {len(extra)} 个: {list(sorted(extra))[:10]} ...")
            print("请先修正集合一致性，再运行本脚本。")
            return

    # 集合一致，开始重排并覆盖
    for i in range(len(dfs)):
        name = FILES[i]
        path = os.path.join(DATA_DIR, name)
        df = dfs[i].set_index("sample_id")
        df_aligned = df.reindex(base)

        if df_aligned.isna().any().any():
            na_rows = df_aligned.isna().any(axis=1)
            print(f"[错误] {name} 对齐后出现 NaN 行：{df_aligned[na_rows].index.tolist()[:10]} ...")
            print("可能有重复 sample_id 或隐式空格，请先清洗数据。")
            return

        # 覆盖前先备份
        bak_path = path + ".bak"  # 生成bak备份
        if not os.path.exists(bak_path):
            os.rename(path, bak_path)
            print(f"[备份] {name} → {os.path.basename(bak_path)}")
        else:
            print(f"[警告] 已存在 {bak_path}，本次将直接覆盖 {name}")

        # 保存覆盖
        df_aligned.reset_index().to_csv(path, index=False)
        print(f"[完成] {name} 已按基准顺序覆盖")

    print("\n 四个文件已对齐并覆盖原文件，可以重新运行训练。")

if __name__ == "__main__":
    main()
