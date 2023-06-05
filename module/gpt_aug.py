import pandas as pd
from tqdm import tqdm

from transformers import GPT2LMHeadModel
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoTokenizer

import torch


#kogpt 모델, 토크나이저 불러오기
model_checkpoint = 'skt/kogpt2-base-v2'
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_data(data, label_dict):
    """
    전처리를 진행하는 함수

    inputs:
        data, label_dict

    returns:
        전처리 적용된 데이터
    """
    concated = []				
    for i in range(len(data)):	
        label = label_dict[data['label'][i] + 1]
        concated.append(label +'SEP' + str(data['text'][i]) + 'EOS')	#라벨이 포함된 텍스트로 전처리를 해줍니다.
    pre_df = pd.DataFrame({'data' : concated})	#전처리된 데이터로 데이터프레임을 만듭니다.

    return pre_df


def tokenize_function(examples):
    result = tokenizer(examples["data"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def group_texts(examples):
    """
    batch마다 들어가는 텍스트의 길이를 통일하는 함수
    """
    chunk_size = 64
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def get_augmentation_data(data, model):
    """
    학습한 모델을 바탕으로 증강을 진행하는 함수

    inputs:
        원본 데이터, 모델

    returns:
        증강된 데이터만 있는 데이터프레임
    """

    data = data.dropna(axis=0) # 결측치 제거

    prompts = list(data['target'])

    for i in range(len(prompts)):
        text = data['text'][i].split(" ")
        prompts[i] = str(prompts[i]) + "SEP" + text[0] + " " + text[1]
    
    created_text = []
    for i in tqdm(range(len(prompts))):
        tokenized_ex = tokenizer.encode(prompts[i], return_tensors='pt')
        gen_ids = model.generate(tokenized_ex,
                                max_length=30,
                                do_sample=True,
                                top_p = 1.0,
                                top_k = 50,
                                repetition_penalty = 1.0,
                                no_repeat_ngram_size=0,
                                #stopping_criteria=['EOS'],
                                temperature=1.0
                                )
        generated = tokenizer.decode(gen_ids[0])
        generated = generated.split('EOS')
        created_text.append(generated[0].split('SEP')[1])

    data['text'] = created_text

    return data


def gptmodel_aug(data):
    """
    gpt 모델을 활용한 증강을 진행하는 함수

    inputs:
        원본 데이터
    
    returns:
        원본 데이터 + gpt 증강 데이터
    """
    df =data.loc[:,['target', 'text']]
    df.columns = ['label', 'text']
    
    #  0 = IT과학,  1= 경제, 2 = 사회, 3 = 생활문화, 4=세계, 5 =스포츠, 6=정치 
    label_dict = {
        1: "it",
        2: "economy",
        3: "social",
        4: "culture",
        5: "world",
        6: "sports",
        7: "policy",
    }

    label_list = label_dict.values()

    df = df.reset_index()

    df = preprocess_data(df, label_dict=label_dict)

    dataset = Dataset.from_pandas(df, split="validtion")
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True) #검증데이터와 훈련데이터로 나눠줍니다.  

    tokenized_datasets = dataset.map(
    tokenize_function, batched = True, remove_columns=["data"]
        )
    
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    batch_size = 16
    model_name = model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=f"{model_name}-gpt_augmentation",
        evaluation_strategy="steps",
        learning_rate=4e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        #fp16=True,
        # logging_steps=logging_steps,
        num_train_epochs=5,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)       

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model("./kogpt")

    model = GPT2LMHeadModel.from_pretrained('./kogpt')

    augment_data = get_augmentation_data(data, model=model)

    all_data = pd.concat([data, augment_data])
    all_data = all_data.sample(frac=1, random_state=42)

    return all_data