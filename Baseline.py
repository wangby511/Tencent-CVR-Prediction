# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""
import sys
import time
import zipfile
import pandas as pd
import numpy as np
from PNN import PNN
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# load data
# data_root = "./data"
# dfTrain = pd.read_csv(data_root + "/train.csv")
# dfTest = pd.read_csv(data_root + "/test.csv")
# dfAd = pd.read_csv(data_root + "/ad.csv")

start = time.time()
# train_user_ad_app = pd.read_csv("train_user_ad_app.csv")
# print("============================read train_user_ad_app.csv FINISH========================")
# end = time.time()
# print(str(end - start) + "s")
#
# test_user_ad_app = pd.read_csv("test_user_ad_app.csv")
# print("============================read test_user_ad_app.csv FINISH========================")
# end = time.time()
# print(str(end - start) + "s")

ignore_features = np.array([
    'conversionTime',
    # 'creativeID',
    'userID',
    'instanceID'
])

numeric_features = np.array([
    'age'
])

one_hot_features = np.array([
    'creativeID', # 6315 个
    'userID', # 2595627 个
    'positionID',
    'connectionType',
    'telecomsOperator',
    'clickTime_hour',
    'gender',
    'education',
    'marriageStatus' ,
    'haveBaby' ,
    'hometown_province',
    'hometown_city',
    'residence_province',
    'residence_city',
    'adID',
    'camgaignID',
    'advertiserID',
    'appID' ,
    'appPlatform' ,
    'app_categories_first_class' ,
    'app_categories_second_class',
    'appCategory'
])

def load_data():
    print("============================load train_user_ad_app.csv & test_user_ad_app.csv========================")
    dfTrain = pd.read_csv("train_user_ad_app.csv")
    dfTest = pd.read_csv("test_user_ad_app.csv")

    df = pd.concat([dfTrain, dfTest],sort=True)

    cols = [c for c in dfTrain.columns if c not in ['label'] and (c not in ignore_features)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['label'].values

    X_test = dfTest[cols].values

    print("============================load csv FINISH========================")

    end = time.time()
    print(str(end - start) + "s")

    return df, dfTrain, dfTest, X_train, y_train, X_test

def split_dimensions(df):
    feat_dict = {}
    tc = 0
    for col in df.columns:
        if col in ignore_features:
            continue

        if col in numeric_features:
            feat_dict[col] = tc
            tc += 1

        else:
            us = df[col].unique()
            feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
            tc += len(us)
            print(len(us),"col =",col,"tc =",tc)
    feat_dimension = tc
    # print (feat_dict)
    print (feat_dimension) # 254
    return feat_dict, feat_dimension

def data_parse(df_data, feat_dict, training = True):

    if training:
        y = df_data['label'].values.tolist()
        df_data.drop(['label'], axis=1, inplace=True)
    else:
        df_data.drop(['label'], axis=1, inplace=True)

    df_index = df_data.copy()
    for col in df_data.columns:
        if col in ignore_features:
            df_data.drop(col, axis = 1, inplace = True)
            df_index.drop(col, axis = 1, inplace = True)
            continue
        if col in numeric_features:
            df_index[col] = feat_dict[col]
        else:
            df_index[col] = df_data[col].map(feat_dict[col])
            df_data[col] = 1.

    xi = df_index.values.tolist()
    xd = df_data.values.tolist()

    if training:
        return xi, xd, y
    return xi, xd

def main():
    df, dfTrain, dfTest, X_train, y_train, X_test = load_data()

    feat_dict, feat_dimension = split_dimensions(df)
    #
    Xi_train, Xv_train, y_train = data_parse(dfTrain, feat_dict, training=True)
    print("Parse Training data done")
    #
    Xi_test, Xv_test = data_parse(dfTest, feat_dict, training=False)
    print("Parse Testing data done")


    # print("Parse Testing data done")
    #
    print(dfTrain.dtypes)
    # creativeID float64
    # positionID float64
    # connectionType float64
    # telecomsOperator float64
    # clickTime_day float64
    # clickTime_hour float64
    # age int64                  age是唯一非one-hot编码的feature
    # gender float64
    # education float64
    # marriageStatus float64
    # haveBaby float64
    # hometown_province float64
    # hometown_city float64
    # residence_province float64
    # residence_city float64
    # adID float64
    # camgaignID float64
    # advertiserID float64
    # appID float64
    # appPlatform float64
    # appCategory float64
    # app_categories_first_class float64
    # app_categories_second_class float64
    # dtype: object

    feature_size = feat_dimension
    field_size = len(Xi_train[0])
    #
    print(feature_size, field_size)  # 17994 23
    #
    pnn_model = PNN(
        feature_size=feat_dimension,
        field_size=len(Xi_train[0]),
        batch_size=128,
        epoch=1
    )
    #
    pnn_model.fit(Xi_train, Xv_train, y_train)

    predict_results = pnn_model.predict(Xi_test, Xv_test)

    print("predict_results =",predict_results)

    predict_results = np.array(predict_results).reshape(-1)

    end = time.time()
    print(str(end - start) + "s")
    print("PREDICT DONE!")

    # # submission
    dfTest = pd.read_csv("test_user_ad_app.csv")
    df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": predict_results})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("my_submission.csv", index=False)
    # with zipfile.ZipFile("submission.zip", "w") as fout:
    #     fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
    print("ALL FINISHED!")

# def main2():
#     dfTest = pd.read_csv("test_user_ad_app.csv")
#     N = len(dfTest)
#     print("N =",N)
#     predict_results = np.zeros([N,1])
#     predict_results = predict_results.reshape(-1)
#
#     df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": predict_results})
#     df.sort_values("instanceID", inplace=True)
#     df.to_csv("my_submission.csv", index=False)


if __name__ == '__main__':
    main()

# end = time.time()
# print(str(end - start) + "s")
# test_features = ['appCategory']
# for feature in test_features:
#     print("train data" + feature +":\n")
#     print(train_user_ad_app[feature].value_counts())
#     print("\n")
#     print("test data" + feature + ":\n")
#     print(test_user_ad_app[feature].value_counts())
#     print("\n")
# # process data
# dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
# dfTest = pd.merge(dfTest, dfAd, on="creativeID")
# y_train = dfTrain["label"].values
#
# # print(dfTrain.columns)
# # ['label', 'clickTime', 'conversionTime', 'creativeID', 'userID','positionID', 'connectionType', 'telecomsOperator', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform']
#
# # feature engineering/encoding
# enc = OneHotEncoder()
# feats = ["creativeID", "adID", "camgaignID", "advertiserID", "appID", "appPlatform"] #  ID类 类别型 ONE-HOT ENCODER
# for i,feat in enumerate(feats):
#     x_train = enc.fit_transform(dfTrain[feat].values.reshape(-1, 1))
#     x_test = enc.transform(dfTest[feat].values.reshape(-1, 1))
#     if i == 0:
#         X_train, X_test = x_train, x_test
#     else:
#         X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
#
# # model training
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# proba_test = lr.predict_proba(X_test)[:,1]
# print(lr.predict_proba(X_test))
#
# # submission
# df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
# df.sort_values("instanceID", inplace=True)
# df.to_csv("submission.csv", index=False)
# with zipfile.ZipFile("submission.zip", "w") as fout:
#     fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)