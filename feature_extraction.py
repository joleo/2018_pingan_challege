# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_extraction
   Description :
   Author :       liyang
   date：          2018/4/17 0017
-------------------------------------------------
   Change Activity:
                   2018/4/17 0017:
-------------------------------------------------
"""
__author__ = 'liyang'

import pandas as pd
import numpy as np
import time
from config import *

class FeatureExtraction(object):
    """
    特征提取
    """

    def read_data(self, path):
        data = pd.read_csv(path)
        return data


    def read_numcial_data(self, data):
        """
        数值型特征
        """
        numcial_column = ['member_id','loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
                   'installment', 'annual_inc', 'dti', 'pub_rec', 'revol_bal', 'revol_util',
                   'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                   'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                   'collection_recovery_fee', 'collections_12_mths_ex_med', 'policy_code',
                   'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim','acc_now_delinq']
        numcial_data = data[numcial_column]

        return numcial_data

    def read_catagory_data(self, data):
        """
        类别型原始特征
        连续字符特征
        1. emp_title 职业，有很多，做特征的时候可以取部分
        2. issue_d 贷款的月份
        3. title 借款人提供的贷款头衔
        4. zip_code 借款人在贷款申请中提供的前3个邮政编码
        5. earliest_cr_line 指借款人的信用额度开通最早的时间
        """
        catagory_column = ['term', 'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership', 'verification_status',
         'issue_d','loan_status', 'pymnt_plan', 'purpose', 'title', 'zip_code', 'addr_state',
         'earliest_cr_line','initial_list_status', 'application_type']

        catagory_column = data[catagory_column]

        return catagory_column

    def get_id(self, data):
        member_id = data['member_id']
        return member_id

    def get_label(self, data):
        acc_now_delinq = data['acc_now_delinq']
        return acc_now_delinq

    def feature_one_hot(self, data):
        """
        one-hot

        [term ，grade，sub_grade，emp_length，home_ownership，verification_status
        loan_status, pymnt_plan purpose, addr_state, initial_list_status ,application_type]
        由于类别太多，sub_grade和addr_state没有one-hot
        """
        dummies_term = pd.get_dummies(data['term'], prefix='term')
        dummies_grade = pd.get_dummies(data['grade'], prefix='grade')
        dummies_emp_length = pd.get_dummies(data['emp_length'], prefix='emp_length')
        dummies_home_ownership = pd.get_dummies(data['home_ownership'], prefix='home_ownership')
        dummies_verification_status = pd.get_dummies(data['verification_status'], prefix='verification_status')
        dummies_loan_status = pd.get_dummies(data['loan_status'], prefix='loan_status')
        dummies_pymnt_plan = pd.get_dummies(data['pymnt_plan'], prefix='pymnt_plan')
        dummies_purpose = pd.get_dummies(data['purpose'], prefix='purpose')
        dummies_addr_initial_list_status = pd.get_dummies(data['initial_list_status'], prefix='initial_list_status')
        dummies_addr_application_type = pd.get_dummies(data['application_type'], prefix='application_type')

        df = pd.concat(
            [dummies_term, dummies_grade, dummies_emp_length, dummies_home_ownership, dummies_verification_status,
             dummies_loan_status, dummies_pymnt_plan, dummies_purpose, dummies_addr_initial_list_status,
             dummies_addr_application_type] ,axis=1)

        return df

    def feature_label_encoder(self, data):
        """
        sub_grade和addr_state没有one-hot
        """
        from sklearn.preprocessing import LabelEncoder
        category_column = ['term' ,'grade','emp_length','home_ownership','verification_status',
                            'loan_status', 'pymnt_plan', 'purpose', 'purpose', 'initial_list_status' ,
                            'application_type']
        category_column = data[category_column].apply(LabelEncoder().fit_transform)

        return category_column

    def loan_amnt_funded_amnt_rate(self, data):
        """
        member_id: LC分配Id
        loan_amnt: 借款人申请贷款的金额
        funded_amnt:在那个时候承诺的贷款总额
        funded_amnt_inv:在那个时间点投资者为该贷款承担的总金额
        installment:  贷款发生时借款人每月支付的款项                 结合时间来做特征
        annual_inc: 借款人在注册期间提供的自我报告的年收入。
        revol_bal: 总信用循环余额
        out_prncp: 剩余未偿还本金总额
        out_prncp_inv: 剩余未偿还本金用于投资者资金总额的一部分
        total_pymnt: 迄今收到的付款总额资金
        total_pymnt_inv: 迄今收到的付款总额的一部分由投资者资助
        total_rec_prncp: 目前收到的本金
        total_rec_int: 迄今收到的利息                       利息应该怎么用呢?
        tot_coll_amt: 总欠款总额
        tot_cur_bal: 所有帐户的当前总余额
        total_rev_hi_lim: 循环信用额度/信用额度总额

        """
        numcial_column = ['member_id', 'loan_amnt', 'funded_amnt',
                                    'funded_amnt_inv', 'installment', 'annual_inc',
                                    'revol_bal','out_prncp','out_prncp_inv','total_pymnt',
                                    'total_pymnt_inv','total_rec_prncp','total_rec_int',
                                    'tot_coll_amt','tot_cur_bal','total_rev_hi_lim']
        numcial_data = data.loc[:, numcial_column]
        # 在那个时候承诺的贷款总额/借款人申请贷款的金额 的比例
        numcial_data.loc[:, 'loan_rate'] = numcial_data['funded_amnt'] / numcial_data['loan_amnt']
        # 在那个时间点投资者为该贷款承担的总金额/ 借款人申请贷款的金额比例
        numcial_data.loc[:, 'loan_timepoint_rate'] = numcial_data['funded_amnt_inv'] / numcial_data['loan_amnt']
        # 年收入 / 借款人申请贷款的金额比例
        numcial_data.loc[:, 'inc_loan_rate'] = numcial_data['annual_inc'] / numcial_data['loan_amnt']
        # 总信用循环余额 / 借款人申请贷款的金额比例
        numcial_data.loc[:, 'revol_bal_loan_rate'] = numcial_data['revol_bal'] / numcial_data['loan_amnt']
        # 剩余未偿还本金总额 / 借款人申请贷款的金额比例
        numcial_data.loc[:, 'out_prncp_loan_rate'] = numcial_data['out_prncp'] / numcial_data['loan_amnt']
        # 剩余未偿还本金用于投资者资金总额的一部分 / 借款人申请贷款的金额比例
        numcial_data.loc[:, 'out_prncp_inv_loan_rate'] = numcial_data['out_prncp_inv'] / numcial_data['loan_amnt']
        # 迄今收到的付款总额资金/ 借款人申请贷款的金额比例
        numcial_data.loc[:, 'pymnt_loan_rate'] = numcial_data['total_pymnt'] / numcial_data['loan_amnt']
        # 迄今收到的付款总额的一部分由投资者资助 / 借款人申请贷款的金额比例
        numcial_data.loc[:, 'pymnt_inv_loan_rate'] = numcial_data['total_pymnt_inv'] / numcial_data['loan_amnt']
        # 总欠款总额 /  借款人申请贷款的金额比例
        numcial_data.loc[:, 'coll_amt_loan_rate'] = numcial_data['tot_coll_amt'] / numcial_data['loan_amnt']
        # 所有帐户的当前总余额 /  借款人申请贷款的金额比例
        numcial_data.loc[:, 'cur_bal_loan_rate'] = numcial_data['tot_cur_bal'] / numcial_data['loan_amnt']
        # 循环信用额度/  借款人申请贷款的金额比例
        numcial_data.loc[:, 'rev_hi_loan_rate'] = numcial_data['total_rev_hi_lim'] / numcial_data['loan_amnt']
        numcial_data.drop(numcial_column, axis=1, inplace=True)

        return numcial_data

    def revol_rate(self, data):
        """
        revol_bal: 总信用循环余额
        revol_util: 循环线使用率或借款人相对于所有可用循环信贷的信用额度
        total_acc;目前在借款人信用档案中的信用额度总数
        """
        revol_ratel_column = ['revol_bal','revol_util','total_acc']
        revol_data = data.loc[:, revol_ratel_column]
        revol_data.loc[:, 'revol_util_bal_rate'] = revol_data['revol_util'] / revol_data['revol_bal']
        revol_data.loc[:, 'total_acc_bal_rate'] = revol_data['total_acc'] / revol_data['revol_bal']
        revol_data.drop(revol_ratel_column, axis=1, inplace=True)

        return revol_data

    # def out_prncp_inv_rate(self, data):
    #     """
    #     out_prncp:剩余未偿还本金总额
    #     out_prncp_inv:剩余未偿还本金用于投资者资金总额的一部分
    #     """
    #     out_prncp_inv_column = ['out_prncp','out_prncp_inv']
    #     out_prncp_inv_data = data.loc[:, out_prncp_inv_column]
    #     out_prncp_inv_data['out_prncp'].map(lambda x: x+1)
    #     out_prncp_inv_data.loc[:, 'out_prncp_inv_rate'] = out_prncp_inv_data['out_prncp_inv'] / out_prncp_inv_data['out_prncp']
    #     out_prncp_inv_data.drop(out_prncp_inv_column, axis=1, inplace=True)
    #
    #     return out_prncp_inv_data

    def total_pymnt_inv_rate(self, data):
        """
        total_pymnt:迄今收到的付款总额为资金
        total_pymnt_inv:迄今收到由投资者资助的付款总额的一部分
        total_rec_prncp:目前收到的本金
        total_rec_int:迄今收到的利息
        """
        total_column = ['total_pymnt', 'total_pymnt_inv','total_rec_prncp', 'total_rec_int']
        total_data = data.loc[:, total_column]
        # 迄今收到由投资者资助的付款总额的一部分/ 迄今收到的付款总额为资金
        total_data.loc[:, 'total_pymnt_inv_rate'] = total_data['total_pymnt_inv'] / total_data[ 'total_pymnt']
        # 迄今收到的利息 /目前收到的本金
        total_data.loc[:, 'total_rec_int_rate'] = total_data['total_rec_int'] / total_data['total_rec_prncp']
        # 目前收到的本金 / 迄今收到的付款总额为资金
        total_data.loc[:, 'total_rec_prncp_pymnt_rate'] = total_data['total_rec_prncp'] / total_data['total_pymnt']
        # 迄今收到的利息 /  迄今收到的付款总额为资金
        total_data.loc[:, 'total_rec_int_pymnt_rate'] = total_data['total_rec_int'] / total_data['total_pymnt']
        total_data.drop(total_column, axis=1, inplace=True)

        return total_data

    """
   #由于属性需要取值种类太多。我们将所有替换为几个关键的属性
   
   若只跟某几个名字有关那么我们需要虚拟化扩维  
   """

    def term_numcial(self, data):
        """
        将贷款的支付次数 数值化
        :param data:
        :return:
        """
        term_data = data.loc[:, 'term']
        term_data.map(lambda x: x.split(' ')[0]).astype(int)

        return term_data

    def cal_loan_rate(self,data):
        # Calculate Good and Bad Loan Status Ratio
        good_loan = len(data[(data.loan_status == 'Fully Paid') |
                           (data.loan_status == 'Current') |
                           (data.loan_status == 'Does not meet the credit policy. Status:Fully Paid')])
        # print('Good/Bad Loan Ratio: %.2f%%' % (good_loan / len(data) * 100))
        loan_rate = good_loan / len(data)
        return loan_rate





