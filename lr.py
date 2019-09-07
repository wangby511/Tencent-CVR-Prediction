import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# # model training
# X_train = [[1,2,3],[-1,-2,-3],[3,2,1]]
# y_train = [1,0,1]
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# X_test = [[-4,-5,-6]]
#
# res = lr.predict_proba(X_test)
# # print(res)
#
#
# enc = OneHotEncoder()
# a = np.array([[3089],
#  [3089],
#  [3089],
#  [6210],
#  [ 310],
#  [ 203]])
# a = a.reshape(-1, 1)
# print (a)
# print (enc.fit_transform(a))
#
# b = np.array([[6210],
#  [203],
#  [3089],
#  [6210],
#  [ 310],
#  [ 203]])
# print ('enc.transform(b) =',enc.transform(b))
feature_labels = np.array(['creativeID',
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
                                         'app_categories_second_class'
                ])


indices = np.array([ 1, 15,  5,  7, 16,  2, 13, 14,  9,  6,  4, 10,  0, 17, 11, 12, 18,  8, 19, 23, 20,  3 ,22 ,21])
for index in indices:
 print(feature_labels[index])
