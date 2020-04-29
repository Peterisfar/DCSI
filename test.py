import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib


def get_score(lr, x_test):
    # 进行模型的预测
    y_pred = lr.predict_proba(x_test)
    return y_pred[:,1]


def DCSI_to_mDCSI(dcsi, m):
    h, w = dcsi.shape
    for col in range(w):
        dcsi_col = dcsi[:, col]

        idx = np.argsort(-dcsi_col)
        dcsi[idx[m:], col] = 0.0

    return dcsi


if __name__ == '__main__':

    data_path = "./data/new"
    weight_path = "train_model.m"
    save_path = "./data/new/mask"


    mask_files = os.listdir(os.path.join(save_path))
    img_files = os.listdir(os.path.join(data_path, "image"))
    classier = joblib.load("train_model.m")


    for i, img_f in enumerate(img_files):
        print(i, img_f)

        if img_f in mask_files:
            print("skip  ", img_f)
            continue

        img = cv2.imread(os.path.join(data_path, "image", img_f))

        h, w, _ = img.shape
        score = np.zeros((h,w), np.float)

        for i in range(h):
            for j in range(w):
                s = [i, j]

                area = np.zeros((16, 16, 3), np.uint8)

                left_x = s[0] - 7 if s[0] - 7 >= 0 else 0
                left_y = s[1] - 7 if s[1] - 7 >= 0 else 0
                right_x = s[0] + 8 if s[0] + 8 < h else h - 1
                right_y = s[1] + 8 if s[1] + 8 < w else w - 1

                crop = img[left_x:right_x + 1, left_y:right_y + 1]
                area[7 - (s[0] - left_x):7 + (right_x - s[0]) + 1, 7 - (s[1] - left_y):7 + (right_y - s[1]) + 1] = crop

                gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
                gray = (gray / 127.5 - 1).flatten()
                score[i, j] = get_score(classier, gray.flatten().reshape(1,-1))

        # cv2.imshow("score", score)
        # cv2.waitKey(0)
        mdcsi = DCSI_to_mDCSI(score, 50)
        cv2.imwrite(os.path.join(save_path, img_f), mdcsi*255.0)