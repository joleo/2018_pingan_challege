# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_preprocess
   Description :
   Author :       liyang
   date：          2018/4/18 0018
-------------------------------------------------
   Change Activity:
                   2018/4/18 0018:
-------------------------------------------------
"""
__author__ = 'liyang'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import *

class DataPreprocess(object):
    """
    数据清洗
    """
    def read_data(self, path):
        data = pd.read_csv(path)
        return data

    def toNumericFromMonth(self, month):
        if type(month) is not float:
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthsval1, monthsval2 = month.split("-")
            ans = months.index(monthsval1) + int(monthsval2) * 12
            return ans
        else:
            return 0

    def removeMonths(self, month):

        return str(month).split(" ")[1]

    def extractNums(self, mixed):
        return filter(str.isdigit, str(mixed))

    def feature_label_encoder(self, data):
        """
        LabelEncoder
        """
        categorical_columns = [ 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'pymnt_plan',
                            'loan_status', 'purpose', 'initial_list_status']
        category_feaure = data[categorical_columns].apply(LabelEncoder().fit_transform)

        return category_feaure

    def data_wash(self, train_df, test_df):
        """
        数据清洗
        """
        # Combine into data:
        train_df['source'] = 'train'
        test_df['source'] = 'test'

        data = pd.concat([train_df, test_df], ignore_index=True)

        # Mapping emp_length
        emp_length_mapping = {'< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4, '4 years': 5, '5 years': 6,
                              '6 years': 7, '7 years': 8, '8 years': 9, '9 years': 10, '10 years': 11, '10+ years': 12}
        data['emp_length'] = data['emp_length'].map(emp_length_mapping)
        data['emp_length'] = data['emp_length'].fillna(0)
        # 提取月份
        data['term_updated'] = pd.to_numeric(data['term'].apply(lambda x: self.removeMonths(x)))
        month_columns = ["earliest_cr_line"]
        for month in month_columns:
            print("Encoding " + month)
            data[month + '_updated'] = data[month].apply(lambda x: self.toNumericFromMonth(x))

        categorical_feature = self.feature_label_encoder(data)

        # 删除不要的特征
        drop_elements = ['open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
                         'total_bal_il','inq_fi', 'total_cu_tl', 'inq_last_12m', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
                         'all_util', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'term',
                         'earliest_cr_line', 'addr_state', 'application_type', 'purpose', 'grade', 'home_ownership', 'sub_grade',
                         'verification_status','pymnt_plan', 'loan_status', 'initial_list_status', 'issue_d', 'zip_code', 'desc', 'emp_title',
                         'title']
        data.drop(drop_elements, axis=1, inplace=True)
        all_feature = pd.concat([data, categorical_feature], axis=1)

        train = all_feature.loc[all_feature['source'] == 'train']
        test = all_feature.loc[all_feature['source'] == 'test']

        train.drop('source', axis=1, inplace=True)
        test.drop(['source', 'acc_now_delinq'], axis=1, inplace=True)

        train.to_csv('data/train_modifier.csv', index=False)
        test.to_csv('data/test_modifier.csv', index=False)


    def down_sample(self, df):
        """
        欠采样
        """
        df1 = df[df['acc_now_delinq'] == 1]
        df2 = df[df['acc_now_delinq'] == 0]
        df3 = df2.sample(frac=0.1)
        return pd.concat([df1, df3], ignore_index=True)


if  __name__ == '__main__':
    dp = DataPreprocess()
    train = dp.read_data(train_path)
    test = dp.read_data(test_path)

    # 数据清洗
    dp.data_wash(train, test)

    # 欠采样
    train_data= dp.read_data('data/train_modifier.csv')
    train_down_data = dp.down_sample(train_data)
    train_down_data.to_csv(train_down_path, index=False)
