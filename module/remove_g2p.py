import requests
import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm


error_id = []
success_id = []


def get_title(id, org_title, url):
    """
    원본 기사 제목을 가져오는 함수

    inputs:
        id, org_title, url

    returns:
        원본 기사 제목(org_title)
    """
    global error_id
    global success_id

    response = requests.get(url, headers={'User-Agent':'Mozilla/5.0'})
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            title = soup.select_one('meta[property="og:title"]')['content']
        except:
            print("Can't get title : ", id, url)
            error_id.append(id)
            return org_title
        else:
            success_id.append(id)
            return title
    else : 
        print("response status code : ", response.status_code)
        print(id, org_title)
        error_id.append(id)
        return org_title
    

def remove_g2p_noise(df):
    """
    데이터에서 g2p 노이즈가 있는, 즉 원본 기사 제목과 text값이 다른 데이터들을 원본 기사 제목으로 대체하는 함수

    inputs:
        원본 dataframe

    returns:
        g2p 노이즈 제거된 dataframe
    """

    print(f"\n\nBefore remove g2p noise : {len(df)}")
    # url이 중복되는 데이터 분리
    noise_df = df[df.duplicated(['ID', 'url'], keep=False)] # noise가 있는 중복된 data
    clean_df = df[~df.duplicated(['ID', 'url'], keep=False)] # noise가 없는 data
    print(f"Noise data : {len(noise_df)}")
    print(f"Clean data : {len(clean_df)}")

    g2p = noise_df[noise_df.duplicated(['ID', 'label_text'], keep='first')]
    print(f"g2p noise data : {len(g2p)}")

    titles = []
    for i, item in tqdm(g2p.iterrows(), total=g2p.shape[0]):
        titles.append(get_title(i, item['input_text'], item['url']))

    g2p['input_text'] = titles
    df = pd.concat([g2p, clean_df])
    print(f"After remove g2p noise : {len(df)}")

    return df
