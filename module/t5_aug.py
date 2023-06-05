import pandas as pd
from tqdm import tqdm

from transformers import AutoModelWithLMHead, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("lcw99/t5-base-korean-paraphrase")
model = AutoModelWithLMHead.from_pretrained("lcw99/t5-base-korean-paraphrase")

def t5_paraphrase(text):
    """
    text를 입력받으면 유사한 문장을 생성해주는 함수

    inputs:
        text
    
    returns:
        paraphrase된 text
    """
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids, num_return_sequences=5, num_beams=5, max_length=128, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    return preds[0]


def t5model_aug(data):
    """
    paraphrasing task로 fine tuning된 t5 모델을 활용하여 데이터를 증강하는 함수

    inputs:
        원본 데이터
    
    returns:
        원본 데이터 + 증강 적용된 데이터
    """
    aug_data = data
    aug_train = []

    for text in tqdm(aug_data['text']):
        aug_train.append(t5_paraphrase("paraphrase: "+text))

    aug_data['text'] = aug_train

    new_data = pd.concat([data, aug_data])
    new_data = new_data.sample(frac=1, random_state=42)

    return new_data
