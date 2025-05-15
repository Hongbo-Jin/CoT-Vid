import os
import argparse
import csv
import json
import pandas as pd

qa_file="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/TempCompass/multi-choice/test-00000-of-00001.parquet"
task_name="TempCompass"
video_root="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/TempCompass/videos/"

YN_qa_file="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/TempCompass/yes_no/test-00000-of-00001.parquet"
def main():
    data_list_info = []
    data=pd.read_parquet(qa_file)
    
    for index,row in data.iterrows():
        # print(row.video_id)
        question=row.question.split('\n')
        candidates=question[1:]
        question=question[0]
        answer_number=candidates.index(row.answer)
        
        data_list_info.append({
                "task_name": task_name+"-"+row.dim,
                "video_name": video_root+row.video_id+".mp4",
                "question_id": 0,
                "question": row.question,
                "answer_number": answer_number,
                "candidates": candidates,
                "answer": row.answer,
            })

    folder = f"/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/{task_name}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/val_mc_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)

def main2():
    data_list_info = []
    data=pd.read_parquet(YN_qa_file)
    
    for index,row in data.iterrows():
        question=row.question
        candidates=['A. yes','B. no']
        answer=row.answer
        if answer=="yes":
            answer_number=0
        elif answer=="no":
            answer_number=1
        else:
            print('wrong') 
            
        data_list_info.append({
                "task_name": task_name+"-"+row.dim,
                "video_name": video_root+row.video_id+".mp4",
                "question_id": 0,
                "question": question,
                "answer_number": answer_number,
                "candidates": candidates,
                "answer": answer,
            })

    folder = f"/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/{task_name}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/val_yes_no_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)


if __name__ == "__main__":
    main()
    main2()
