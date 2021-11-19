"""
Assume df is a pandas dataframe object of the dataset given
"""

import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the entire dataset'''


def calculate_entropy(positive, negative):
    total = positive + negative
    if positive == 0:
        entropy = - (negative / total) * np.log2(negative / total)
    elif negative == 0:
        entropy = - (positive / total) * np.log2(positive / total)
    else:
        entropy = (-positive / total) * np.log2(positive / total) - (negative / total) * np.log2(negative / total)
    return entropy


# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    target = df.iloc[:, -1]
    p = 0
    n = 0
    for sample in target:
        if sample == 'yes':
            p += 1
        elif sample == 'no':
            n += 1
    entropy = calculate_entropy(p, n)
    return entropy


'''Return avg_info of the attribute provided as parameter'''


# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    column_df = pd.DataFrame(df[attribute])
    column_df['target'] = df.iloc[:, -1]
    avg_info = 0
    positives = {}
    negatives = {}
    for entry in range(len(column_df[attribute])):
        value = column_df.iloc[entry][attribute]
        if value not in positives:
            positives[value] = 0
            negatives[value] = 0
        if column_df.iloc[entry]['target'] == 'yes':
            positives[value] += 1
        else:
            negatives[value] += 1
    total_p = 0
    total_n = 0
    for key in positives:
        p = positives[key]
        n = negatives[key]
        total_p += p
        total_n += n
        avg_info += (p + n) * calculate_entropy(p, n)

    avg_info = avg_info / (total_p + total_n)
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''


# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    information_gain = get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)
    return round(information_gain, 7)


# input: pandas_dataframe
# output: ({dict},'str')
def get_selected_attribute(df):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    """
    ig = {}
    for column in df.iloc[:, :-1]:
        ig[column] = get_information_gain(df, column)
    max_gain = max(ig, key=lambda x: ig[x])
    return ig, max_gain
