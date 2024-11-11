import sqlite3
import numpy as np
from enum import Enum
# from typing import List


# SCHEMA: TABLE data (videoID, Video_Name, Layer_3, Layer_4, AvgPool, BOF_HOG, BOF_HOF, Action_Label);

connection = sqlite3.connect('../database/Phase_2.db')
c = connection.cursor()

# valid_column_names = enumerate(['id', 'name', '1', '2', '3', '4', '5', 'label'])
class column(Enum):
    id = 'videoID'
    name = 'Video_Name'
    '1' = 'Layer_3'
    '2' = 'Layer_4'
    '3' = 'AvgPool'
    '4' = 'BOF_HOG'
    '5' = 'BOF_HOF'

Feature_Space = ['Layer_3', 'Layer_4', 'AvgPool', 'BOF_HOG', 'BOF_HOF']
# Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}


# def queryGenerator(columnsList: List[column], condition: str):
def queryGenerator(columnsList: list[column], condition: str):
    columns = ", ".join(columnsList)
    feature_index = (i for i, x in enumerate(columnsList) if column[x] in Feature_Space)

    execution_query = f"SELECT {columns} FROM data {condition};"
    return execution_query, feature_index

def queryDB(columns: list[column], condition: str):
    execution_query, f_index = queryGenerator(columns, condition)
    
    c.execute(execution_query)
    rows = c.fetchall()
    
    cleaned_data = []

    if columns[f_index] in ['Layer_3', 'Layer_4', 'AvgPool']:
        for row in rows:
            cleaned_data.append(list(map(float, row[f_index].strip("[]").split(","))))
    else:
        for row in rows:
            cleaned_data.append(list(map(int, row[f_index].strip("[]").split())))

    max_len = max(len(lst) for lst in cleaned_data)
    padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
    data = np.array(padded_data)
    return data

# Get Action_Label


# connection.close()
