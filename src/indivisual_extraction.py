import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# 各自のデータから動作時のみ抽出


def linear_acc(df):
    # 重力加速度を取り除くための関数(androidのアルゴリズムを流用)
    # 初期値はそのまま重力加速度とする
    alpha = 0.8
    gravity_x = df.iat[0, 0]
    gravity_y = df.iat[0, 1]
    gravity_z = df.iat[0, 2]
    df.iat[0, 0] = df.iat[0, 0] - gravity_x
    df.iat[0, 1] = df.iat[0, 1] - gravity_y
    df.iat[0, 2] = df.iat[0, 2] - gravity_z

    for i in range(1, df.shape[0]):
        gravity_x = alpha * gravity_x + (1 - alpha) * df.iat[i, 0]
        gravity_y = alpha * gravity_y + (1 - alpha) * df.iat[i, 1]
        gravity_z = alpha * gravity_z + (1 - alpha) * df.iat[i, 2]

        df.iat[i, 0] = df.iat[i, 0] - gravity_x
        df.iat[i, 1] = df.iat[i, 1] - gravity_y
        df.iat[i, 2] = df.iat[i, 2] - gravity_z
    return df


def extraction(df, add_file):
    start_count = 0
    end_count = 0
    start_point = False
    end_point = False
    for i in range(0, df.shape[0]):
        # gyro_magがしきい値35を25回超えたら動作開始，下回ったら動作終了
        # しきい値はheuristicに設定
        if (start_point == False) & (df["gyro_mag"].iloc[i,] >= 35):
            start_count += 1
            if start_count == 25:
                start = i - 50
                start_point = True
                start_count = 0
        else:
            start_count = 0

        if (start_point == True) & (df["gyro_mag"].iloc[i,] <= 35):
            end_count += 1
            if end_count == 25:
                end = i + 25
                end_point = True
                end_count = 0
        else:
            end_count = 0

        if end_point == True:
            df_extraction = df.iloc[start : end + 1,]
            if df_extraction.shape[0] >= 100:
                df_extraction.to_csv(
                    "../extraction_csv/normal/"
                    + df["id"].iloc[0,]
                    + "/"
                    + str(add_file)
                    + ".csv",
                    index=False,
                )
                add_file += 1
            start_point = False
            end_point = False
    return add_file


def main():
    # データフレーム読み込み
    add_file = 0
    for file in range(1, 8):
        # df = pd.read_csv('../csv/raw_csv/user' + args.f + '/' + str(file) + '.csv', header = None)
        df = pd.read_csv(
            "../sensor_data/raw_csv/user" + args.f + "/" + str(file) + ".csv",
            header=None,
        )

        # df = pd.read_csv(args.f + '/' + str(file) + '.csv')
        df.columns = [
            "accX",
            "accY",
            "accZ",
            "gyroX",
            "gyroY",
            "gyroZ",
            "magX",
            "magY",
            "magZ",
            "angleX",
            "angleY",
            "angleZ",
            "time",
            "id",
        ]

        # 格納するフォルダがなければ作成
        if not os.path.exists("../extraction_csv"):
            os.makedirs("../extraction_csv")
        if not os.path.exists("../extraction_csv/collab_spoof"):
            os.makedirs("../extraction_csv/collab_spoof")
        if not os.path.exists("../extraction_csv/normal/" + df["id"].iloc[0]):
            os.makedirs("../extraction_csv/normal/" + df["id"].iloc[0])

        # 加速度から重力加速度の寄与を取り除き線形加速度に
        df = linear_acc(df)

        # l2normを算出
        df["acc_mag"] = np.sqrt(df["accX"] ** 2 + df["accY"] ** 2 + df["accZ"] ** 2)
        df["gyro_mag"] = np.sqrt(df["gyroX"] ** 2 + df["gyroY"] ** 2 + df["gyroZ"] ** 2)
        add_file = extraction(df, add_file)
        print(add_file)


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
