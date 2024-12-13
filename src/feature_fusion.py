import pandas as pd
import numpy as np
from tsfresh import extract_features
from sklearn.preprocessing import StandardScaler
from scipy import signal
import sys
import os
import argparse

# 特徴抽出

i = "related_feature"


# directory作成
def create_directory():
    if not os.path.exists("../pair_feature/" + i):
        os.makedirs("../pair_feature/" + i)
    if not os.path.exists("../pair_feature/" + i + "/" + args.f2 + "_" + args.f1):
        os.makedirs("../pair_feature/" + i + "/" + args.f2 + "_" + args.f1)


# sklearnによる標準化
def standardization(df):

    standard = StandardScaler().fit_transform(df)
    standard_df = pd.DataFrame(standard)

    return standard_df


def corr_feature(features, df, name):
    corr_matrix = df[[name + "X", name + "Y", name + "Z"]].corr()

    features[name + "X_Y__corr"] = corr_matrix.iat[1, 0]
    features[name + "Y_Z__corr"] = corr_matrix.iat[2, 1]
    features[name + "Z_X__corr"] = corr_matrix.iat[2, 0]

    return features


# 特徴抽出関数
def features_ext(df, header_names):
    features = extract_features(
        df,
        # df[column_names],
        column_id="id",
        column_kind=None,
        column_value=None,
        default_fc_parameters={
            "maximum": None,
            "minimum": None,
            "median": None,
            "sample_entropy": None,
            "skewness": None,
        },
        kind_to_fc_parameters=None,
    )

    selected_columns = [
        f"{header}__{stat}"
        for header in header_names
        for stat in ["maximum", "minimum", "median", "sample_entropy", "skewness"]
    ]

    features = features[selected_columns]

    return features, selected_columns


def main():

    # フォルダ内の全てのcsvファイルを読み込む
    create_directory()

    # 特徴量格納用のデータフレームを用意
    if args.f1 == "normal":
        file_range = range(1, 7)
    elif args.f1 == "mastery":
        file_range = range(100, 140)
    else:
        file_range = range(0, 70)

    for pair in range(0, 1):
        features_df = pd.DataFrame()
        for file in file_range:
            # フォルダ内のcsvを読み込み
            df = pd.read_csv(
                "../pair_extraction_csv/"
                + args.f1
                + "/"
                + str(pair)
                + "_0_"
                + args.f1
                + "/"
                + str(file)
                + ".csv",
                index_col=False,
            )
            df2 = pd.read_csv(
                "../pair_extraction_csv/"
                + args.f1
                + "/"
                + str(pair)
                + "_1_"
                + args.f1
                + "/"
                + str(file)
                + ".csv",
                index_col=False,
            )

            # sklearnによる標準化
            # 特徴抽出する時系列データを選択
            header_names = [
                "accX",
                "accY",
                "accZ",
                "gyroX",
                "gyroY",
                "gyroZ",
                "angleX",
                "angleY",
                "angleZ",
                "acc_mag",
                "gyro_mag",
            ]
            drop_df = df[header_names]
            drop_df2 = df2[header_names]

            sc_df = standardization(drop_df)
            sc_df2 = standardization(drop_df2)

            sc_df.columns = header_names
            sc_df2.columns = header_names

            # args.f2がdata_meanを指定していたらポイントごとにデータの平均をとって統合
            if args.f2 == "data_mean":
                DataFusion_df = (sc_df + sc_df2) / 2
                DataFusion_df["id"] = pd.DataFrame(df["id"])

                # data統合したDFから特徴抽出
                fusion_df, selected_columns = features_ext(DataFusion_df, header_names)
                fusion_df = fusion_df.reset_index(drop=True)

            # args.f2がdata_normを指定していたらポイントごとにデータの二乗和平方根をとって統合
            elif args.f2 == "data_norm":
                DataFusion_df = pd.DataFrame()
                for col in header_names:
                    DataFusion_df[col] = np.sqrt(sc_df[col] ** 2 + sc_df2[col] ** 2)
                DataFusion_df["id"] = pd.DataFrame(df["id"])

                # data統合したDFから特徴抽出
                fusion_df, selected_columns = features_ext(DataFusion_df, header_names)
                fusion_df = fusion_df.reset_index(drop=True)

            # そのほかだったら特徴統合
            else:
                # 各DFから特徴抽出
                sc_df["id"] = pd.DataFrame(df["id"])
                sc_df2["id"] = pd.DataFrame(df2["id"])
                features, selected_columns = features_ext(sc_df, header_names)
                features2, selected_columns = features_ext(sc_df2, header_names)
                features = features.reset_index(drop=True)
                features2 = features2.reset_index(drop=True)
                # args.f2がfeature_meanを指定していたら特徴ごとに平均をとって統合
                if args.f2 == "feature_mean":
                    fusion_df = (features + features2) / 2
                # そのほかだったらfeature_normとして特徴ごとに二乗和平方根をとって統合
                else:
                    fusion_df = pd.DataFrame()
                    for col in selected_columns:
                        fusion_df[col] = np.sqrt(
                            features[col] ** 2 + features2[col] ** 2
                        )

            features_df = pd.concat([features_df, fusion_df], axis=0)
            features_df.index.name = "label"

        # 特徴量を保存したdfをcsvで保存
        features_df.to_csv(
            "../pair_feature/"
            + i
            + "/"
            + args.f2
            + "_"
            + args.f1
            + "/"
            + str(pair)
            + ".csv",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Interval_extraction",  # プログラム名
        usage="Enter csv filename",  # プログラムの利用方法
        description="description",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument("f1", help="file_name", type=str)
    parser.add_argument("f2", help="fusion_name", type=str)
    args = parser.parse_args()
    main()
