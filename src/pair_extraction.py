import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# ペアのデータから動作時のみ抽出


def create_directory():
    if not os.path.exists("../pair_extraction_csv/collab_spoof/"):
        os.makedirs("../pair_extraction_csv/collab_spoof/")
    if not os.path.exists("../pair_extraction_csv/normal/" + args.f + "_0/"):
        os.makedirs("../pair_extraction_csv/normal/" + args.f + "_0/")
    if not os.path.exists("../pair_extraction_csv/normal/" + args.f + "_1/"):
        os.makedirs("../pair_extraction_csv/normal/" + args.f + "_1/")


def extraction(df):
    threshold = 2
    start_count = 0
    end_count = 0
    start_comp = False
    for i in range(0, df.shape[0]):
        if (start_comp == False) & (df["gyro_mag"].iloc[i,] >= 35):
            start_count += 1
            if start_count == 20:
                start_point = i - 19
                start_comp = True
        else:
            start_count = 0
        if (start_comp == True) & (df["gyro_mag"].iloc[i,] <= 35):
            end_count += 1
            if end_count == 20:
                end_point = i - 19
                break
        else:
            end_count = 0
    return start_point, end_point


def pair_extraction(df, df2, file):
    start, end = extraction(df)
    start2, end2 = extraction(df2)

    start_time1 = df["time"].iloc[start,]
    start_time2 = df2["time"].iloc[start2,]
    end_time1 = df["time"].iloc[end,]
    end_time2 = df2["time"].iloc[end2,]

    value = 1000000
    if (
        (start_time1.hour > start_time2.hour)
        | (start_time1.minute > start_time2.minute)
        | (start_time1.second > start_time2.second)
        | (
            (
                start_time1.strftime("%Y-%m-%d %H:%M:%S")
                == start_time2.strftime("%Y-%m-%d %H:%M:%S")
            )
            & (start_time1.microsecond > start_time2.microsecond)
        )
    ):
        for i in range(start2, df2.shape[0]):
            if start_time1.strftime("%Y-%m-%d %H:%M:%S") == df2["time"].iloc[
                i,
            ].strftime("%Y-%m-%d %H:%M:%S"):
                if (
                    abs(df2["time"].iloc[i,].microsecond - start_time1.microsecond)
                    < value
                ):
                    value = abs(
                        df2["time"].iloc[i,].microsecond - start_time1.microsecond
                    )
                    start2 = i
                else:
                    break

    elif (
        (start_time1.hour < start_time2.hour)
        | (start_time1.minute < start_time2.minute)
        | (start_time1.second < start_time2.second)
        | (
            (
                start_time1.strftime("%Y-%m-%d %H:%M:%S")
                == start_time2.strftime("%Y-%m-%d %H:%M:%S")
            )
            & (start_time1.microsecond < start_time2.microsecond)
        )
    ):
        for i in range(start, df.shape[0]):
            if df["time"].iloc[i,].strftime(
                "%Y-%m-%d %H:%M:%S"
            ) == start_time2.strftime("%Y-%m-%d %H:%M:%S"):
                if (
                    abs(df["time"].iloc[i,].microsecond - start_time2.microsecond)
                    < value
                ):
                    value = abs(
                        df["time"].iloc[i,].microsecond - start_time2.microsecond
                    )
                    start = i
                else:
                    break

    value = 1000000
    if (
        (end_time1.hour > end_time2.hour)
        | (end_time1.minute > end_time2.minute)
        | (end_time1.second > end_time2.second)
        | (
            (
                end_time1.strftime("%Y-%m-%d %H:%M:%S")
                == end_time2.strftime("%Y-%m-%d %H:%M:%S")
            )
            & (end_time1.microsecond > end_time2.microsecond)
        )
    ):
        for i in range(df.shape[0] - 1, start, -1):
            if df["time"].iloc[i,].strftime("%Y-%m-%d %H:%M:%S") == end_time2.strftime(
                "%Y-%m-%d %H:%M:%S"
            ):
                if abs(df["time"].iloc[i,].microsecond - end_time2.microsecond) < value:
                    value = abs(df["time"].iloc[i,].microsecond - end_time2.microsecond)
                    end = i
                else:
                    break

    elif (
        (end_time1.hour < end_time2.hour)
        | (end_time1.minute < end_time2.minute)
        | (end_time1.second < end_time2.second)
        | (
            (
                end_time1.strftime("%Y-%m-%d %H:%M:%S")
                == end_time2.strftime("%Y-%m-%d %H:%M:%S")
            )
            & (end_time1.microsecond < end_time2.microsecond)
        )
    ):
        for i in range(df2.shape[0] - 1, start2, -1):
            if end_time1.strftime("%Y-%m-%d %H:%M:%S") == df2["time"].iloc[i,].strftime(
                "%Y-%m-%d %H:%M:%S"
            ):
                if (
                    abs(df2["time"].iloc[i,].microsecond - end_time1.microsecond)
                    < value
                ):
                    value = abs(
                        df2["time"].iloc[i,].microsecond - end_time1.microsecond
                    )
                    end2 = i
                else:
                    break

    df = df.iloc[start:end,]
    df2 = df2.iloc[start2:end2,]
    return df, df2


def pair_uni(df, df2):
    if df.shape[0] > df2.shape[0]:
        df = df.iloc[0 : df2.shape[0],]
    elif df.shape[0] < df2.shape[0]:
        df2 = df2.iloc[0 : df.shape[0], :]
    return df, df2


def main():
    for file in range(1, 7):
        file_name = "../extraction_csv/normal/" + args.f + "_0/" + str(file) + ".csv"
        file_name2 = "../extraction_csv/normal/" + args.f + "_1/" + str(file) + ".csv"
        df = pd.read_csv(file_name, header=0)
        df2 = pd.read_csv(file_name2, header=0)
        df["time"] = df["time"].astype(str)
        df2["time"] = df2["time"].astype(str)
        df["time"] = pd.to_datetime(df["time"])
        df2["time"] = pd.to_datetime(df2["time"])
        create_directory()
        df, df2 = pair_extraction(df, df2, file)
        df, df2 = pair_uni(df, df2)
        df.to_csv(
            "../pair_extraction_csv/normal/" + args.f + "_0/" + str(file) + ".csv",
            index=False,
        )
        df2.to_csv(
            "../pair_extraction_csv/normal/" + args.f + "_1/" + str(file) + ".csv",
            index=False,
        )
        if abs(df.shape[0] - df2.shape[0]) != 0:
            print(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Interval_extraction",  # プログラム名
        usage="Enter csv filename",  # プログラムの利用方法
        description="description",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument("f", help="file_name", type=str)
    args = parser.parse_args()
    main()
