import os
import argparse
import yaml

import pandas as pd

from module.base_preprocessing import base_ppc
from module.remove_g2p import remove_g2p_noise
from module.gpt_aug import gptmodel_aug
from module.t5_aug import t5model_aug
from module.spelling_clean import clean_spell
from module.batch_stratify import batch_stratify



if __name__ == "__main__":

    # 파일 이름을 받고, 어떤 전처리 작업을 진행할지 입력받습니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--file_name', type=str, default="train")
    args = parser.parse_args()

    with open('./config.yaml', encoding='UTF8') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    FILE_NAME = args.file_name
    print("file_name : ", FILE_NAME)

    # 파일 이름에 해당하는 csv 파일을 불러옵니다.(전처리 대상)
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, './data')
    
    data = pd.read_csv(os.path.join(DATA_DIR, FILE_NAME + '.csv'))


    # 불러온 파일에 전처리 작업을 진행한 후, 저장합니다.
    METHOD_NAME=""
    for method in CFG['preprocessing']:
        preprocessed_data = globals()[method](data)
        METHOD_NAME += "_" + method

    preprocessed_data.to_csv(os.path.join(DATA_DIR, FILE_NAME + METHOD_NAME + '.csv'))