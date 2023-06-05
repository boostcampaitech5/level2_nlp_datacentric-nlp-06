import os
import random
import numpy as np
import pandas as pd
import torch
import evaluate
import pickle
import re

def remove_unwanted_chars(text):
    """
    특수문자 제거 함수

    Returns :
        특수문제가 제거된 문장
    """
    # load
    with open('./module/symbols.pickle', 'rb') as f:
        data = pickle.load(f)
    symbols = set(data)
    
    # Define the regular expression pattern for Korean, English, and specified symbols
    pattern = re.compile(r"[^a-zA-Zㄱ-ㅎㅣ가-힣一-\uFFFF" + re.escape("".join(symbols)) + r"0-9 ]")
    
    # Use the pattern to remove unwanted characters
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text


def shorten(clean_data):
    """
    데이터셋에서 sports label에 속하는 일부 문장들은 전체 문장을 사용하지 않고 "..."으로 축약되어 사용되는 것을 확인
    -> 이를 맞춰주기 위해 sports label에 속하고 문장의 길이가 특정 값 이상이면 축약을 적용하는 함수

    Returns :
        축약이 적용된 dataframe 
    """
    new_processed = []
    for i,d in clean_data.iterrows():
        if d['url'].find('sports')!=-1 and d['text'].find('...')== -1 and len(d['text'])>36:
            d['text'] = d['text'][:34]+'...'
        new_processed.append(d['text'])
    clean_data['text'] = new_processed
    return clean_data
    
    
def base_ppc(df):
    """
    base preprocessing 과정을 진행하는 함수

    Returns:
        base preprocessing이 적용된 dataframe
    """
    df['text'] = df['text'].apply(lambda row: remove_unwanted_chars(row))
    df = shorten(df)
    
    return df