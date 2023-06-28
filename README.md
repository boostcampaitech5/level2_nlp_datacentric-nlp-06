# 1. 프로젝트 개요
# 1.1 개요
KLUE-Topic Classification benchmark는 뉴스의 헤드라인을 통해 그 뉴스가 어떤 topic을 갖는지를 분류해내는 task입니다. 인공지능의 성능을 향상시키기 위해 데이터의 품질을 개선하는 것에 집중하는 접근인 Data-Centric 의 취지에 맞게, 베이스라인 모델의 수정 없이 오로지 데이터의 수정으로만 성능 향상을 이끌어내는 것이 이 프로젝트의 목표입니다.
## 1.2 평가 지표
Macro F1 score

# 2. 프로젝트 팀 구성 및 역할

## 2.1. 팀 구성
|<img src='https://avatars.githubusercontent.com/u/74442786?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/99644139?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/50359820?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/85860941?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/106165619?v=4' height=100 width=100px></img>|
|:---:|:---:|:---:|:---:|:---:|
| [김민호](https://github.com/GrapeDiget) | [김성은](https://github.com/seongeun-k) | [김지현](https://github.com/jihyeeon) | [서가은](https://github.com/gaeun0112) | [홍영훈](https://github.com/MostlyFor) |

## 2.2. 역할
- 김민호 : 프로젝트 리팩토링, 외부 데이터 사용 증강, 경계값 제거
- 김성은 : 프로젝트 리팩토링, G2P 노이즈 클리닝, 요약 모델 사용 증강, SMOTE, chatGPT API 사용 증강
- 김지현 : 프로젝트 리팩토링, Base 전처리, Stop word 제거, Masking, Batch stratify, key terms 활용 전처리 및 증강
- 서가은 : back translation, paraphrasing, gpt 모델 활용 데이터 증강
- 홍영훈 : Label Error Detection, 데이터 증강, Key terms masking

# 3. Data Analysis

- 원본 데이터
  - 총 학습 데이터는 45,678개이며 이 데이터 중 15%는 의도적으로 노이즈가 추가된 데이터. 노이즈 데이터 중 80%는 G2P를 이용하여 문장을 소리나는 대로 바꾼 데이터이며, 20%는 labeling 오류로 생성된 데이터. 
- 데이터 분포를 살펴 본 결과 데이터 수가 적은 label인 경우 잘 맞추지 못하는 경향이 있음. -> 데이터 수가 적은 label 증강 필요.

# 4. 프로젝트 수행 결과 
## 4.1. Data Filtering & Cleaning
- G2P 노이즈
  - 주어진 url을 통해 기사 제목을 스크래핑하여 G2P 노이즈 데이터 클리닝.
- Label Error Detection
  - 기존 데이터와 외부 데이터를 cleanlab library를 이용하여 label error를 탐지, 제거.
- Masking
  - 특정 단어의 중요성을 줄이기 위해 자주 나오는 단어를 masking 처리하여 학습.
- Stop word 제거
  - 데이터 셋에 자주 등장하지만 큰 의미를 갖지 않는 단어인 stop word 를 제거.
- 맞춤법 정제
  - 맞춤법 오류가 있는 문장들을 교정하기 위해 hanspell 라이브러리를 사용하여 맞춤법 정제.
- 경계값 제거
  - Label별 예측 확률을 통해 가장 큰 값과 그 다음 값의 차이가 크지 않은 데이터를 제거한 후 재학습 진행.
- Base 전처리
  - 증강 또는 클리닝을 수행한 데이터가 원본 데이터와 유사한 형식을 유지할 수 있도록 데이터 정제
## 4.2. Data Augmentation
- BT & IBT(Back Translation & Iterative Back Translation)
  - papago & pororo: 다양한 방식으로 input text 에 Back Translation 과 Iterative Back Translation 을 수행하여 데이터를 증강.
- GPT 모델 사용
  - Pre-Trained GPT 모델을 대회 데이터로 fine tuning하여 Label을 input을 넣으면 text를 생성해주는 모델로 학습한 후, 이를 이용하여 증강.
- 요약 모델 사용
  - 주어진 url을 통해 뉴스 기사 본문을 추출한 후 huggingface의 csebuetnlp/mT5_multilingual_XLSum 모델을 활용하여 요약한 텍스트를 데이터로 활용. 
- Paraphrasing
  - Huggingface의 lcw99/t5-base-korean-paraphrase모델을 활용하여 text와 다르지만 의미적으로는 유사한 문장들을 생성하여 데이터로 활용.
## 4.3. 외부 데이터 사용
- chatGPT 증강
  - chatGPT API를 통해 레이블마다 선정된 key terms에 관한 뉴스 기사 제목 생성.
- AIHub & 한국언론진흥재단 뉴스빅데이터
  - 오픈소스 기사 데이터를 활용하여 기사의 제목과 기사 분류를 사용.
## 4.4. 기타
- Label sampling 
  - 증강 데이터 활용 시 모델이 잘 예측하지 못하는 레이블에 대하여 샘플링하여 활용. 
- Batch stratify
  - 데이터셋이 Shuffle 되지 않는 프로젝트의 특성을 고려하여 하나의 batch 가 모든 Label 을 포함할 수 있도록 데이터셋을 조정.
- SMOTE
  - 데이터 불균형을 해결하기 위하여 개수가 적은 레이블에 대한 데이터를 임의로 생성하는 오버샘플링 기법 활용.
- Key terms 활용
  - Label 별 Key terms 를 활용하여 다양한 방식으로 input text 에 추가, 이를 통해 Label 에 대한 추가 정보를 모델에 제공하고자 함. (전처리 / 증강)
