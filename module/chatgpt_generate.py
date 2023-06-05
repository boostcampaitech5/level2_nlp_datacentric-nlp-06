import openai
import pandas as pd
import time


def get_content(query):
    try : 
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}])
    except:
        print("openai error")
        print("sleeping 100s...")
        time.sleep(100)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}])
    else:
        print("generate success")
        completion = completion.choices[0].message.content
    
    return completion

def main():
    openai.api_key = "Your OpenAI api key" # API Key

    key_terms = {'정치' : ['청와대', '대통령실', '정부', '국회', '의회', '북한', '정당', '국방', '외교', '행정']}
            #  '경제' : ['주식', '금융', '산업', '재계', '기업', '부동산'],
            #  '사회' : ['교육', '노동', '언론', '환경', '인권', '식품', '의약품'],
            #  '문화' : ['건강', '교통', '레저', '핫플레이스', '패션', '뷰티', '공연', '전시', '책', '날씨'],
            #  '세계' : ['아시아', '호주', '미주', '유럽', '중동', '아프리카'],
            #  'IT' : ['모바일', 'IT', '인터넷' ,'소셜 미디어', '커뮤니케이션']}
            #  '스포츠' : ['야구', '농구', '배구']}
    
    terms = []
    generated = []

    label = '정치'
    for term in key_terms[label]:
        terms.append(term)
        generate_query = f"{term}에 대한 한국 뉴스 기사 제목을 50개 만들어줘"
        generated.append(get_content(generate_query))

    data = {'term':terms, 'generated':generated}

    df = pd.DataFrame(data)
    df.to_csv(f"../data/chatgpt_generated_{label}.csv", index=False)
