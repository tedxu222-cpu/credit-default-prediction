"""信贷违约预测项目：数据预处理完整流程。

该脚本用于从原始数据 `credit_data.csv` 生成清洗后的
`credit_data_cleaned.csv`。处理逻辑与第二阶段 notebook 保持一致：

1. 删除重复记录。
2. 删除已确认需要删除的非匿名字段缺失记录。
3. 将 employmentLength 缺失值保留为 Unknown 类别。
4. 保留 n0-n14 匿名字段原始缺失值，不做主观插补。
5. 删除业务逻辑异常的 dti 负值记录。
6. 仅按 IQR 规则删除 revolUtil 的异常极端值。
7. 检查字符串与日期字段一致性。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "isDefault"
RAW_DATA_PATH = Path("credit_data.csv")
CLEAN_DATA_PATH = Path("credit_data_cleaned.csv")


DROP_MISSING_COLS = [
    "employmentTitle",
    "postCode",
    "title",
    "revolUtil",
    "pubRecBankruptcies",
    "dti",
]

ANONYMOUS_COLS = [f"n{i}" for i in range(15)]

OUTLIER_CHECK_COLS = [
    "loanAmnt",
    "interestRate",
    "installment",
    "annualIncome",
    "dti",
    "delinquency_2years",
    "ficoRangeLow",
    "ficoRangeHigh",
    "openAcc",
    "pubRec",
    "pubRecBankruptcies",
    "revolBal",
    "revolUtil",
    "totalAcc",
]

VALID_EMPLOYMENT_LENGTHS = {
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
    "Unknown",
}


def load_data(input_path: Path) -> pd.DataFrame:
    """读取原始数据。"""
    if not input_path.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{input_path}")
    return pd.read_csv(input_path)


def calculate_iqr_bounds(series: pd.Series) -> tuple[float, float, float, float, float]:
    """计算 IQR 下界和上界。"""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return q1, q3, iqr, lower_bound, upper_bound


def build_business_logic_summary(data: pd.DataFrame) -> pd.DataFrame:
    """汇总业务逻辑异常数量，用于清洗前后核查。"""
    checks: list[dict[str, object]] = []

    positive_cols = ["loanAmnt", "term", "interestRate", "installment"]
    non_negative_cols = ["annualIncome", "dti", "revolBal", "revolUtil"]
    non_negative_integer_cols = [
        "id",
        "employmentTitle",
        "purpose",
        "postCode",
        "regionCode",
        "delinquency_2years",
        "ficoRangeLow",
        "ficoRangeHigh",
        "openAcc",
        "pubRec",
        "pubRecBankruptcies",
        "totalAcc",
        "homeOwnership",
        "verificationStatus",
        "initialListStatus",
        "applicationType",
        "title",
        "policyCode",
        *[column for column in ANONYMOUS_COLS if column in data.columns],
    ]

    def add_check(name: str, columns: list[str], abnormal_func) -> None:
        existing_cols = [column for column in columns if column in data.columns]
        abnormal_count = 0
        abnormal_fields: list[str] = []
        for column in existing_cols:
            series = data[column].dropna()
            count = int(abnormal_func(series).sum())
            if count > 0:
                abnormal_fields.append(column)
            abnormal_count += count
        checks.append(
            {
                "检查项": name,
                "检查字段": existing_cols,
                "异常记录数": abnormal_count,
                "异常字段": abnormal_fields,
            }
        )

    add_check("必须大于0", positive_cols, lambda series: series <= 0)
    add_check("必须大于或等于0", non_negative_cols, lambda series: series < 0)
    add_check(
        "必须为非负整数",
        non_negative_integer_cols,
        lambda series: (series < 0) | (series % 1 != 0),
    )

    if {"ficoRangeLow", "ficoRangeHigh"}.issubset(data.columns):
        checks.append(
            {
                "检查项": "FICO下限必须小于或等于上限",
                "检查字段": ["ficoRangeLow", "ficoRangeHigh"],
                "异常记录数": int((data["ficoRangeLow"] > data["ficoRangeHigh"]).sum()),
                "异常字段": ["ficoRangeLow", "ficoRangeHigh"],
            }
        )

    if "term" in data.columns:
        checks.append(
            {
                "检查项": "term应为3或5",
                "检查字段": ["term"],
                "异常记录数": int((~data["term"].isin([3, 5])).sum()),
                "异常字段": ["term"],
            }
        )

    if "policyCode" in data.columns:
        checks.append(
            {
                "检查项": "policyCode应为1或2",
                "检查字段": ["policyCode"],
                "异常记录数": int((~data["policyCode"].isin([1, 2])).sum()),
                "异常字段": ["policyCode"],
            }
        )

    return pd.DataFrame(checks)


def check_string_and_date_consistency(data: pd.DataFrame) -> pd.DataFrame:
    """检查字符串、等级、工作年限和日期字段的一致性。"""
    checks: list[dict[str, object]] = []

    string_cols = [
        column
        for column in ["grade", "subGrade", "employmentLength", "issueDate", "earliesCreditLine"]
        if column in data.columns
    ]

    for column in string_cols:
        values = data[column].astype("string")
        checks.append(
            {
                "检查项": f"{column} 空字符串",
                "异常数量": int(values.str.len().eq(0).sum()),
            }
        )
        checks.append(
            {
                "检查项": f"{column} 首尾空格",
                "异常数量": int(values.ne(values.str.strip()).sum()),
            }
        )

    if {"grade", "subGrade"}.issubset(data.columns):
        checks.append(
            {
                "检查项": "subGrade首字母与grade一致",
                "异常数量": int(data["subGrade"].str[0].ne(data["grade"]).sum()),
            }
        )

    if "employmentLength" in data.columns:
        checks.append(
            {
                "检查项": "employmentLength取值在预期范围内",
                "异常数量": int((~data["employmentLength"].isin(VALID_EMPLOYMENT_LENGTHS)).sum()),
            }
        )

    issue_date = pd.to_datetime(data["issueDate"], format="%Y-%m-%d", errors="coerce")
    earlies_credit_line = pd.to_datetime(
        data["earliesCreditLine"], format="%b-%Y", errors="coerce"
    )

    checks.extend(
        [
            {"检查项": "issueDate可解析", "异常数量": int(issue_date.isna().sum())},
            {
                "检查项": "issueDate为每月1日",
                "异常数量": int((issue_date.dt.day != 1).sum()),
            },
            {
                "检查项": "earliesCreditLine可解析",
                "异常数量": int(earlies_credit_line.isna().sum()),
            },
            {
                "检查项": "earliesCreditLine不晚于issueDate",
                "异常数量": int((earlies_credit_line > issue_date).sum()),
            },
        ]
    )

    return pd.DataFrame(checks)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """执行第二阶段确定的数据清洗流程。"""
    df_clean = df.copy()
    summary: dict[str, object] = {
        "原始行数": len(df_clean),
        "原始列数": df_clean.shape[1],
        "原始缺失单元格数": int(df_clean.isna().sum().sum()),
    }

    duplicate_count = int(df_clean.duplicated().sum())
    df_clean = df_clean.drop_duplicates().copy()
    summary["删除重复行数"] = duplicate_count

    missing_row_mask = df_clean[DROP_MISSING_COLS].isna().any(axis=1)
    deleted_missing_rows = int(missing_row_mask.sum())
    df_clean = df_clean.loc[~missing_row_mask].copy()
    summary["删除指定缺失字段行数"] = deleted_missing_rows

    employment_missing_count = int(df_clean["employmentLength"].isna().sum())
    df_clean["employmentLength"] = df_clean["employmentLength"].fillna("Unknown")
    summary["employmentLength填补Unknown数量"] = employment_missing_count

    negative_dti_count = int((df_clean["dti"] < 0).sum())
    df_clean = df_clean.loc[df_clean["dti"] >= 0].copy()
    summary["删除dti负值行数"] = negative_dti_count

    revolutil_series = df_clean["revolUtil"].dropna()
    q1, q3, iqr, lower_bound, upper_bound = calculate_iqr_bounds(revolutil_series)
    revolutil_outlier_mask = (df_clean["revolUtil"] < lower_bound) | (
        df_clean["revolUtil"] > upper_bound
    )
    deleted_revolutil_outliers = int(revolutil_outlier_mask.sum())
    df_clean = df_clean.loc[~revolutil_outlier_mask].copy()

    summary.update(
        {
            "revolUtil_Q1": q1,
            "revolUtil_Q3": q3,
            "revolUtil_IQR": iqr,
            "revolUtil_IQR下界": lower_bound,
            "revolUtil_IQR上界": upper_bound,
            "删除revolUtil极端值行数": deleted_revolutil_outliers,
            "清洗后行数": len(df_clean),
            "清洗后列数": df_clean.shape[1],
            "清洗后缺失单元格数": int(df_clean.isna().sum().sum()),
        }
    )

    return df_clean, summary


def validate_clean_data(df_clean: pd.DataFrame, summary: dict[str, object]) -> None:
    """对清洗结果进行一致性检查。"""
    existing_anonymous_cols = [column for column in ANONYMOUS_COLS if column in df_clean.columns]
    non_anonymous_cols = [
        column for column in df_clean.columns if column not in existing_anonymous_cols
    ]

    if df_clean.shape != (798_774, 47):
        raise AssertionError(f"清洗后数据形状异常：{df_clean.shape}")
    if int(df_clean.duplicated().sum()) != 0:
        raise AssertionError("清洗后仍存在重复行")
    if int(df_clean[TARGET].isna().sum()) != 0:
        raise AssertionError("目标字段存在缺失值")
    if int(df_clean[non_anonymous_cols].isna().sum().sum()) != 0:
        raise AssertionError("非匿名字段仍存在缺失值")
    if int(df_clean[existing_anonymous_cols].isna().sum().sum()) != 612_517:
        raise AssertionError("匿名字段缺失数量与预期不一致")
    if int((df_clean["dti"] < 0).sum()) != 0:
        raise AssertionError("dti仍存在负值")
    if float(df_clean["revolUtil"].max()) > float(summary["revolUtil_IQR上界"]):
        raise AssertionError("revolUtil仍存在超出IQR上界的记录")

    business_logic_summary = build_business_logic_summary(df_clean)
    if int(business_logic_summary["异常记录数"].sum()) != 0:
        raise AssertionError("业务逻辑一致性检查未通过")

    consistency_summary = check_string_and_date_consistency(df_clean)
    if int(consistency_summary["异常数量"].sum()) != 0:
        raise AssertionError("字符串或日期一致性检查未通过")


def save_data(df_clean: pd.DataFrame, output_path: Path) -> None:
    """保存清洗后的数据。"""
    df_clean.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="信贷违约预测项目数据预处理脚本")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help="原始数据CSV路径，默认 credit_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLEAN_DATA_PATH,
        help="清洗后数据CSV路径，默认 credit_data_cleaned.csv",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过固定结果一致性检查",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    df_clean, summary = clean_data(df)

    if not args.skip_validation:
        validate_clean_data(df_clean, summary)

    save_data(df_clean, args.output)

    print("数据预处理完成")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
