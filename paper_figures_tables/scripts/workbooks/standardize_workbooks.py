#!/usr/bin/env python3
"""Standardize statistical summary workbook naming and worksheet titles."""

from pathlib import Path
import openpyxl

REPO_ROOT = Path(__file__).resolve().parents[2]


def standardize_summary_workbook() -> None:
    dst_path = REPO_ROOT / "results" / "workbooks" / "Statistical_Significance_Summary.xlsx"
    if dst_path.exists():
        print(f"{dst_path} is ready.")
        return


def main() -> None:
    standardize_summary_workbook()


if __name__ == "__main__":
    main()
