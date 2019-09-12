#coding=utf-8
import time
import sys
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#文件读取
def read_csv_file(f,logging=False):
    print ("============================读取数据========================",f)
    data = pd.read_csv(f)
    if logging:
        print (data.head(5))
        print (f,"  包含以下列....")
        print (data.columns.values)
        # print (data.describe())
        # print (data.info())
    return data

#第一类编码
def categories_process_first_class(cate):
    cate = str(cate)
    # if len(cate) == 1:
    #     if int(cate) == 0:
    #         return 0
    # else:
    return int(cate[0])

#第2类编码
def categories_process_second_class(cate):
    cate = str(cate)
    if len(cate) < 3:
        return 0
    else:
        return int(cate[1:])

#年龄处理，切段
def age_process(age):
    age = int(age)
    if age == 0:
        return 0
    elif age < 15:
        return 1
    elif age < 25:
        return 2
    elif age < 40:
        return 3
    elif age < 60:
        return 4
    else:
        return 5

#省份处理
def get_province(info):
    if info == 0:
        return 0
    info = str(info)
    province = int(info[0:2])
    return province

#城市处理
def get_city(info):
    info = str(info)
    if len(info) > 1:
        city = int(info[2:])
    else:
        city = 0
    return city

#几点钟
def get_time_day(t):
    t = str(t)
    t = int(t[0:2])
    return t

#一天切成4段
def get_time_hour(t):
    t = str(t)
    t = int(t[2:4])
    if t < 6:
        return 0
    elif t < 12:
        return 1
    elif t < 18:
        return 2
    else:
        return 3

#评估与计算logloss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1,act) * sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def pre_processing_data():

    #['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator']
    train_data = read_csv_file('./data/train.csv',logging = False)

    #['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform']
    ad = read_csv_file('./data/ad.csv',logging = False)

    # app
    # ['appID', 'appCategory']
    app_categories = read_csv_file('./data/app_categories.csv', logging = False)
    app_categories["app_categories_first_class"] = app_categories['appCategory'].apply(categories_process_first_class)
    app_categories["app_categories_second_class"] = app_categories['appCategory'].apply(categories_process_second_class)

    # print(app_categories.app_categories_first_class.values)
    # print(app_categories.app_categories_second_class.values)
    # print(app_categories.head())
    # print(app_categories.appCategory.describe())
    # cnt = {}
    # for x in app_categories.appCategory.values:
    #     cnt[x] = cnt.get(x,0) + 1
    # print(cnt)
    # user = read_csv_file('./data/user.csv', logging = False)
    # print(user.columns)
    # Index(['userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence'], dtype='object')
    # print(user[user.age != 0].describe())
    # print("user.age.value_counts() = \n",user.age.value_counts())
    # print("app_categories.appCategory.value_counts() = \n",app_categories.appCategory.value_counts())

    # user
    user = read_csv_file('./data/user.csv', logging=False)
    # ['userID' 'age' 'gender' 'education' 'marriageStatus' 'haveBaby' 'hometown' 'residence']
    # user['age_process'] = user['age'].apply(age_process)
    user["hometown_province"] = user['hometown'].apply(get_province)
    user["hometown_city"] = user['hometown'].apply(get_city)
    user["residence_province"] = user['residence'].apply(get_province)
    user["residence_city"] = user['residence'].apply(get_city)
    # print(user.hometown_province.value_counts())

    # print(user.hometown_city.value_counts())
    # print(user.residence_province.value_counts())
    # print(user.residence_city.value_counts())

    print ("============================user process done========================")
    # print (user.head())
    # print (user.info())
    # print (train_data.head())
    # train data
    train_data['clickTime_day'] = train_data['clickTime'].apply(get_time_day)
    train_data['clickTime_hour'] = train_data['clickTime'].apply(get_time_hour)

    # train_data['conversionTime_day'] = train_data['conversionTime'].apply(get_time_day)
    # train_data['conversionTime_hour'] = train_data['conversionTime'].apply(get_time_hour)

    # test_data
    test_data = read_csv_file('./data/test.csv', True)
    test_data['clickTime_day'] = test_data['clickTime'].apply(get_time_day)
    test_data['clickTime_hour'] = test_data['clickTime'].apply(get_time_hour)
    # test_data['conversionTime_day'] = test_data['conversionTime'].apply(get_time_day)
    # test_data['conversionTime_hour'] = test_data['conversionTime'].apply(get_time_hour)

    print("============================pd.merge training data========================")
    train_user = pd.merge(train_data, user, on='userID')
    train_user_ad = pd.merge(train_user, ad, on='creativeID')
    train_user_ad_app = pd.merge(train_user_ad, app_categories, on='appID')
    print("============================pd.merge training data done========================")

    print(train_user_ad_app.head())
    print(train_user_ad_app.info())
    print(train_user_ad_app.describe())

    print("============================train_user_ad_app.columns========================")
    # Index(['label', 'conversionTime', 'creativeID', 'userID',
    #        'positionID', 'connectionType', 'telecomsOperator', 'clickTime_day',
    #        'clickTime_hour', 'age', 'gender', 'education', 'marriageStatus',
    #        'haveBaby', 'hometown', 'residence', 'age_process', 'hometown_province',
    #        'hometown_city', 'residence_province', 'residence_city', 'adID',
    #        'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory',
    #        'app_categories_first_class', 'app_categories_second_class'],
    #       dtype='object')
    drop_columns = ['clickTime', 'hometown', 'residence']
    train_user_ad_app = train_user_ad_app.drop(columns=drop_columns)
    print(train_user_ad_app.columns)
    train_user_ad_app.to_csv("train_user_ad_app.csv",index=0)

    print("============================pd.merge test data========================")
    test_user = pd.merge(test_data, user, on='userID')
    test_user_ad = pd.merge(test_user, ad, on='creativeID')
    test_user_ad_app = pd.merge(test_user_ad, app_categories, on='appID')

    print("============================pd.merge training data done========================")
    test_user_ad_app = test_user_ad_app.drop(columns=drop_columns)
    test_user_ad_app = test_user_ad_app.sort_values(by=['instanceID'])
    print(test_user_ad_app.columns)
    test_user_ad_app.to_csv("test_user_ad_app.csv", index=0)


    return train_user_ad_app


start = time.time()
train_user_ad_app = pre_processing_data()
end = time.time()
print("============================data processing FINISH========================")
print(str(end - start) + "s")
#特征部分
sys.exit(0)
x_user_ad_app = train_user_ad_app.loc[:,['creativeID',
                                         'userID',
                                         'positionID',
                                         'connectionType',
                                         'telecomsOperator',
                                         'clickTime_day',
                                         'clickTime_hour',
                                         'age',
                                         'gender',
                                         'education',
                                         'marriageStatus',
                                         'haveBaby',
                                         # 'residence',
                                         'age_process',
                                         'hometown_province',
                                         'hometown_city',
                                         'residence_province',
                                         'residence_city',
                                         'adID',
                                         'camgaignID',
                                         'advertiserID',
                                         'appID',
                                         'appPlatform',
                                         'app_categories_first_class',
                                         'app_categories_second_class']
                ]

x_user_ad_app = x_user_ad_app.values
x_user_ad_app = np.array(x_user_ad_app, dtype='int32')

#标签部分
y_user_ad_app =train_user_ad_app.loc[:,['label']].values

feature_labels = np.array(['creativeID',
                           'userID',
                           'positionID',
                           'connectionType',
                           'telecomsOperator',
                           'clickTime_day',
                           'clickTime_hour',
                           'age',
                           'gender',
                           'education','marriageStatus' ,
                           'haveBaby' ,
                           # 'residence' ,
                           'age_process',
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
                           'app_categories_second_class'
                           ])

print("============================Building RF========================")
forest = RandomForestClassifier(n_estimators=100,
                                random_state=0,
                                n_jobs=-1)
forest.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0],))
importances = forest.feature_importances_
print(importances)
indices = np.argsort(importances)[::-1]
print(indices)
#
# [0.0279838  0.16562236 0.0660558  0.00436996 0.03338885 0.07715558
#  0.03894126 0.07583464 0.01295768 0.0493107  0.03123428 0.01825503
#  0.09906992 0.01595917 0.05430419 0.04849448 0.06369797 0.05796617
#  0.01879754 0.01566209 0.00780727 0.006694   0.00076299 0.00298131
#  0.00669295]
# [ 1 12  5  7  2 16 17 14  9 15  6  4 10  0 18 11 13 19  8 20 21 24  3 23 22]

# [0.02947309 0.17758784 0.07170534 0.00449408 0.0361463  0.08316023
#  0.04104759 0.08087065 0.01342539 0.05229715 0.03285699 0.01891242
#  0.01621501 0.06470519 0.05414106 0.08992944 0.0722904  0.01944727
#  0.01605816 0.00912135 0.00560669 0.00086411 0.00297897 0.00666526]
# [ 1 15  5  7 16  2 13 14  9  6  4 10  0 17 11 12 18  8 19 23 20  3 22 21]

end = time.time()
print("============================ALL FINISH========================")
print(str(end - start) + "s")

# feature 重要度排序: 影响力由大到小
# userID
# residence
# clickTime_day
# age
# positionID
# residence_province
# residence_city
# hometown_province
# education
# hometown_city
# clickTime_hour
# telecomsOperator
# marriageStatus
# creativeID
# adID
# haveBaby
# age_process
# camgaignID
# gender
# advertiserID
# appID
# app_categories_second_class
# connectionType
# app_categories_first_class
# appPlatform

#第二遍运行排序 除去residence
# userID
# residence_province
# clickTime_day
# age
# residence_city
# positionID
# hometown_province
# hometown_city
# education
# clickTime_hour
# telecomsOperator
# marriageStatus
# creativeID
# adID
# haveBaby
# age_process
# camgaignID
# gender
# advertiserID
# app_categories_second_class
# appID
# connectionType
# app_categories_first_class
# appPlatform