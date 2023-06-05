import random
import pandas as pd

from sklearn.model_selection import train_test_split


def batch_stratify(data):
    """
    각 batch마다 Label당 데이터들이 균일하게 들어가도록 하는 함수

    inputs:
        전체 데이터

    outputs:
        stratified sampling이 적용된 데이터
    """
    shuffle = True
    # PREFIX = 'Stratified_'
    new_data = data

    # stratified batches 만들기
    batches = []
    while len(new_data)>39:
        train, _ = train_test_split(new_data, train_size=32, stratify=new_data["target"])
        new_data.drop(train.index, inplace=True)
        batches.append(train.sample(frac=1))
    else:
        batches.append(new_data)

    # For shuffling batches
    if shuffle==True:
        random.shuffle(batches)
        # PREFIX = 'Shuffled_'+PREFIX

    new_df = pd.concat(batches, ignore_index=True)
    
    return new_df