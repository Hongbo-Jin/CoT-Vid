import pandas as pd
import json
from pathlib import Path

df = pd.read_csv("perception.csv")
print(df.head())  # 查看前5行
print(df.shape)

data=[]

video_cnt=0
image_cnt=0

choices=[
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

for row in range(df.shape[0]):
    
    video_path="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/VideoMMMU/videos/"+df.iloc[row]['id']+'.mp4'

    path=Path(video_path)
    if not path.exists():
        continue
        image_cnt+=1
    else:
        video_cnt+=1
    
    question=df.iloc[row]['question']
    options=df.iloc[row]['options']
    # options=df.iloc[row]['options'].split(" ")
    # options[0]=options[0][1:]
    # print(options)
    # print(options[0])
    # print(options[1])
    # exit(0)
    answer=df.iloc[row]['answer']
    # if answer<='F' and answer>='A':
    #     pass
    # else:
    #     print(answer)
    question_type=df.iloc[row]['question_type']
    print(len(options))
    if question_type=="multiple-choice":
        data.append({
            "video_path":video_path,
            "question":question,
            "options":options,
            "answer":answer,
        })
    
print(image_cnt)
print(video_cnt)

with open('perception.json','w',encoding='utf-8') as file:
    json.dump(data,file,indent=4)
    
