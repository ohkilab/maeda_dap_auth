import pandas as pd
import numpy as np
from sklearn import svm
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import optuna
import pickle
from scipy import optimize
from scipy import interpolate
import argparse


def rcsv(path, label):
    X = pd.read_csv(path, header=0)
    y = pd.DataFrame()
    for index_num in range(0, X.shape[0]):
        y = y.append({"label": label}, ignore_index=True)
    return X, y


def another_file(
    file,
    X_train,
    X_test,
    y_train,
    y_test,
    X_mastery1,
    X_mastery2,
    y_mastery1,
    y_mastery2,
):
    for another_file in range(1, 8):
        if file != another_file:
            X, y = rcsv(
                "../pair_feature/"
                + args.f1
                + "/"
                + args.f3
                + "_normal/"
                + str(another_file)
                + ".csv",
                0,
            )
            X_mastery, y_mastery = rcsv(
                "../pair_feature/"
                + args.f1
                + "/"
                + args.f3
                + "_mastery/"
                + str(another_file)
                + ".csv",
                0,
            )
            another_X_train, another_X_test = X.iloc[:80,], X.iloc[80:,]
            another_y_train, another_y_test = y.iloc[:80,], y.iloc[80:,]
            another_X_mastery1, another_X_mastery2 = (
                X_mastery.iloc[:20,],
                X_mastery.iloc[20:,],
            )
            another_y_mastery1, another_y_mastery2 = (
                y_mastery.iloc[:20,],
                y_mastery.iloc[20:,],
            )

            X_train = pd.concat([X_train, another_X_train], axis=0)
            X_test = pd.concat([X_test, another_X_test], axis=0)
            y_train = pd.concat([y_train, another_y_train], axis=0)
            y_test = pd.concat([y_test, another_y_test], axis=0)

            X_mastery1 = pd.concat([X_mastery1, another_X_mastery1], axis=0)
            X_mastery2 = pd.concat([X_mastery2, another_X_mastery2], axis=0)
            y_mastery1 = pd.concat([y_mastery1, another_y_mastery1], axis=0)
            y_mastery2 = pd.concat([y_mastery2, another_y_mastery2], axis=0)
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_mastery1,
        X_mastery2,
        y_mastery1,
        y_mastery2,
    )


def y_trans(y):
    y = y.values
    y = y.astype(int)
    y = y.ravel()
    return y


def objective(trial, X_train, y_train, X_test, y_test):
    C = trial.suggest_float("C", 1e-5, 1e5, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e5, log=True)
    clf = svm.SVC(C=C, gamma=gamma, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    fnr = 1 - tpr
    eer1 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer2 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer1 + eer2) / 2
    return eer


def clf_scale(Y_test, pred_proba, file):
    scale_df = pd.DataFrame(columns=["EER"], index=[file])
    fpr, tpr, thresholds = roc_curve(Y_test, pred_proba)
    fnr = 1 - tpr
    eer = optimize.brentq(
        lambda x: 1.0 - x - interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0
    )
    eeridx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer_threshold = thresholds[eeridx]
    scale_df["EER"] = eer

    return scale_df, fpr, tpr, fnr, eer_threshold, thresholds


def clf_score(
    clf, file, X_test, y_test, all_fpr, all_tpr, all_fnr, all_thr, score_df, scores_df
):
    pred_proba = clf.predict_proba(X_test)[:, 1]
    scale_df, fpr, tpr, fnr, eer_threshold, thresholds = clf_scale(
        y_test, pred_proba, file
    )
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_fnr.append(fnr)
    all_thr.append(thresholds)
    score_df = pd.concat([score_df, scale_df], axis=0)
    scores_df = pd.concat([scores_df, score_df], axis=0)
    return all_fpr, all_tpr, all_fnr, all_thr, scores_df


def plot_roc(all_fpr, all_tpr, all_fnr, all_thr):
    mean_fpr = np.unique(np.concatenate([all_fpr[i] for i in range(0, 8)]))
    mean_tpr = np.zeros_like(mean_fpr)

    for i in range(0, 8):
        mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i], left=0, right=1)
    mean_tpr /= 8
    mean_thr = np.zeros_like(mean_fpr)

    for i in range(0, 8):
        mean_thr += np.interp(mean_fpr, all_fpr[i], all_thr[i], left=0, right=1)
    mean_thr /= 8
    mean_fnr = np.zeros_like(mean_fpr)

    for i in range(0, 8):
        mean_fnr += np.interp(mean_fpr, all_fpr[i], all_fnr[i], left=0, right=1)
    mean_fnr /= 8
    mean_tpr = np.insert(mean_tpr, 0, 0)
    mean_fpr = np.insert(mean_fpr, 0, 0)
    # mean_thr = np.insert(mean_thr, 9 , 0)

    auc_macro = auc(mean_fpr, mean_tpr)
    print(auc_macro)

    return mean_fnr, mean_fpr, mean_tpr, mean_thr


def plt_bar(df, img_label):
    ax = df.plot.bar(legend=False)
    ax.tick_params(axis="x", rotation=0)
    plt.xlabel("認証ペア")
    plt.ylabel("EER")
    plt.ylim(0, 1)
    plt.savefig(
        "../img/"
        + args.f2
        + "/"
        + args.f2
        + "_"
        + args.f3
        + "_"
        + img_label
        + "EERs.png"
    )
    plt.show()


def objective(trial, X_train, y_train, X_test, y_test):
    C = trial.suggest_float("C", 1e-5, 1e5, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e5, log=True)
    clf = svm.SVC(C=C, gamma=gamma, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    eer = optimize.brentq(
        lambda x: 1.0 - x - interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0
    )
    return eer


def main():
    scores_df1 = pd.DataFrame()
    scores_df2 = pd.DataFrame()
    scores_df3 = pd.DataFrame()
    scores_df4 = pd.DataFrame()
    scores_df5 = pd.DataFrame()
    all_tpr = []
    all_fpr = []
    all_fnr = []
    all_thr = []
    all_tpr_mastery1 = []
    all_fpr_mastery1 = []
    all_fnr_mastery1 = []
    all_thr_mastery1 = []
    all_tpr_mastery2 = []
    all_fpr_mastery2 = []
    all_fnr_mastery2 = []
    all_thr_mastery2 = []
    all_fpr_spoof = []
    all_tpr_spoof = []
    all_fnr_spoof = []
    all_thr_spoof = []
    all_fpr_collab_spoof = []
    all_tpr_collab_spoof = []
    all_fnr_collab_spoof = []
    all_thr_collab_spoof = []
    optuna.logging.disable_default_handler()
    for file in range(0, 8):
        score_df1 = pd.DataFrame()
        score_df2 = pd.DataFrame()
        score_df3 = pd.DataFrame()
        score_df4 = pd.DataFrame()
        score_df5 = pd.DataFrame()
        X, y = rcsv(
            "../pair_feature/"
            + args.f1
            + "/"
            + args.f3
            + "_normal/"
            + str(file)
            + ".csv",
            1,
        )
        X_mastery, y_mastery = rcsv(
            "../pair_feature/"
            + args.f1
            + "/"
            + args.f3
            + "_mastery/"
            + str(file)
            + ".csv",
            1,
        )
        X_spoof, y_spoof = rcsv(
            "../pair_feature/"
            + args.f1
            + "/"
            + args.f3
            + "_spoof/"
            + str(file)
            + ".csv",
            0,
        )
        X_collab_spoof, y_collab_spoof = rcsv(
            "../pair_feature/"
            + args.f1
            + "/"
            + args.f3
            + "_collab_spoof/"
            + str(file)
            + ".csv",
            0,
        )

        X_train, X_test = X.iloc[:80,], X.iloc[80:,]
        y_train, y_test = y.iloc[:80,], y.iloc[80:,]

        X_spoof = pd.concat([X_test, X_spoof], axis=0)
        y_spoof = pd.concat([y_test, y_spoof], axis=0)

        X_collab_spoof = pd.concat([X_test, X_collab_spoof], axis=0)
        y_collab_spoof = pd.concat([y_test, y_collab_spoof], axis=0)

        X_mastery1, X_mastery2 = X_mastery.iloc[:20,], X_mastery.iloc[20:,]
        y_mastery1, y_mastery2 = y_mastery.iloc[:20,], y_mastery.iloc[20:,]

        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_mastery1,
            X_mastery2,
            y_mastery1,
            y_mastery2,
        ) = another_file(
            file,
            X_train,
            X_test,
            y_train,
            y_test,
            X_mastery1,
            X_mastery2,
            y_mastery1,
            y_mastery2,
        )

        y_train = y_trans(y_train)
        y_test = y_trans(y_test)
        y_mastery1 = y_trans(y_mastery1)
        y_mastery2 = y_trans(y_mastery2)
        y_spoof = y_trans(y_spoof)
        y_collab_spoof = y_trans(y_collab_spoof)

        if args.f2 == "svm":
            clf = svm.SVC(probability=True)
        elif args.f2 == "rf":
            clf = RandomForestClassifier()
        elif args.f2 == "lgbm":
            clf = lgb.LGBMClassifier()
        elif args.f2 == "xgb":
            clf = XGBClassifier()
        else:
            print("モデル名をコマンドライン引数に入れてください")
            print("matching.py filename modelname")
            exit()
        clf.fit(X_train, y_train)

        all_fpr, all_tpr, all_fnr, all_thr, scores_df1 = clf_score(
            clf,
            file,
            X_test,
            y_test,
            all_fpr,
            all_tpr,
            all_fnr,
            all_thr,
            score_df1,
            scores_df1,
        )
        (
            all_fpr_mastery1,
            all_tpr_mastery1,
            all_fnr_mastery1,
            all_thr_mastery1,
            scores_df2,
        ) = clf_score(
            clf,
            file,
            X_mastery1,
            y_mastery1,
            all_fpr_mastery1,
            all_tpr_mastery1,
            all_fnr_mastery1,
            all_thr_mastery1,
            score_df2,
            scores_df2,
        )
        (
            all_fpr_mastery2,
            all_tpr_mastery2,
            all_fnr_mastery2,
            all_thr_mastery2,
            scores_df3,
        ) = clf_score(
            clf,
            file,
            X_mastery2,
            y_mastery2,
            all_fpr_mastery2,
            all_tpr_mastery2,
            all_fnr_mastery2,
            all_thr_mastery2,
            score_df3,
            scores_df3,
        )
        all_fpr_spoof, all_tpr_spoof, all_fnr_spoof, all_thr_spoof, scores_df4 = (
            clf_score(
                clf,
                file,
                X_spoof,
                y_spoof,
                all_fpr_spoof,
                all_tpr_spoof,
                all_fnr_spoof,
                all_thr_spoof,
                score_df4,
                scores_df4,
            )
        )
        (
            all_fpr_collab_spoof,
            all_tpr_collab_spoof,
            all_fnr_collab_spoof,
            all_thr_collab_spoof,
            scores_df5,
        ) = clf_score(
            clf,
            file,
            X_collab_spoof,
            y_collab_spoof,
            all_fpr_collab_spoof,
            all_tpr_collab_spoof,
            all_fnr_collab_spoof,
            all_thr_collab_spoof,
            score_df5,
            scores_df5,
        )

        with open(
            "model/" + args.f2 + "/pair" + str(file) + "_" + args.f3 + "_model.pickle",
            mode="wb",
        ) as f:
            pickle.dump(clf, f, protocol=3)
    plt.figure(figsize=(8, 6))

    plt.style.use("tableau-colorblind10")
    mean_fnr1, mean_fpr1, mean_tpr1, mean_thr1 = plot_roc(
        all_fpr, all_tpr, all_fnr, all_thr
    )
    mean_fnr2, mean_fpr2, mean_tpr2, mean_thr2 = plot_roc(
        all_fpr_mastery1, all_tpr_mastery1, all_fnr_mastery1, all_thr_mastery1
    )
    mean_fnr3, mean_fpr3, mean_tpr3, mean_thr3 = plot_roc(
        all_fpr_mastery2, all_tpr_mastery2, all_fnr_mastery2, all_thr_mastery2
    )
    mean_fnr4, mean_fpr4, mean_tpr4, mean_thr4 = plot_roc(
        all_fpr_spoof, all_tpr_spoof, all_fnr_spoof, all_thr_spoof
    )
    mean_fnr5, mean_fpr5, mean_tpr5, mean_thr5 = plot_roc(
        all_fpr_collab_spoof,
        all_tpr_collab_spoof,
        all_fnr_collab_spoof,
        all_thr_collab_spoof,
    )

    with open("session1.txt", "w") as f:
        print(scores_df1.mean(), file=f)
    with open("session2.txt", "w") as f:
        print(scores_df2.mean(), file=f)
    with open("session3.txt", "w") as f:
        print(scores_df3.mean(), file=f)
    with open("spoof.txt", "w") as f:
        print(scores_df4.mean(), file=f)
    with open("collab-spoof.txt", "w") as f:
        print(scores_df5.mean(), file=f)

    # 平均化したROC曲線をプロット
    plt.plot(mean_fpr1, mean_tpr1, linestyle="-", linewidth=2, label="session1")
    plt.plot(mean_fpr2, mean_tpr2, linestyle="-", linewidth=2, label="session2")
    plt.plot(mean_fpr3, mean_tpr3, linestyle="-", linewidth=2, label="session3")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.savefig("../img/" + args.f2 + "/" + args.f2 + "_" + args.f3 + "_ROC.png")
    plt.show()

    plt.plot(mean_fpr4, mean_tpr4, linestyle="-", linewidth=2, label="spoof")
    plt.plot(mean_fpr5, mean_tpr5, linestyle="-", linewidth=2, label="collab-spoof")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.savefig("../img/" + args.f2 + "/" + args.f2 + "_" + args.f3 + "_ROC_spoof.png")
    plt.show()

    plt_bar(scores_df1, "normal")
    plt_bar(scores_df2, "mastery1")
    plt_bar(scores_df3, "mastery2")
    plt_bar(scores_df4, "spoof")
    plt_bar(scores_df5, "collab_spoof")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Interval_extraction",  # プログラム名
        usage="Enter csv filename",  # プログラムの利用方法
        description="description",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument("f1", help="folder_name", type=str)
    parser.add_argument("f2", help="model_name", type=str)
    parser.add_argument("f3", help="fusion_name", type=str)
    args = parser.parse_args()
    main()
