import os
import argparse
import csv
import json
import pandas as pd
from pathlib import Path
import numpy as np

task_name=" MMVU "
data_dir="/mnt/cloud_disk/public_data/MMVU"
qa_file="/mnt/cloud_disk/public_data/MMVU/validation.json"

choices=["A","B","C","D","E"]

def main():
    data_list_info = []
    with open(qa_file,'r',encoding='utf-8') as file:
        data_info=json.load(file)
    
    oe_cnt=0
    mc_data_list_info=[]
    for item in data_info:
        print(item)
        exit(0)
        question_type=item['question_type']
        if question_type=="open-ended":
            oe_cnt+=1
            continue
        elif question_type=="multiple-choice":
            mc_data_list_info.append(item)
        else:
            print('err')

    print(oe_cnt)
    print(len(mc_data_list_info))
    print(len(data_info))
    exit(0)
    
    df=pd.read_parquet(qa_file)
    for index,row in df.iterrows():
        question_id=row['id']
        dataset=row['dataset']
        scene_name=row['scene_name']
        question_type=row['question_type']
        question=row['question']
        candidates=row['options']
        video_path=data_dir+dataset+'/'+str(scene_name)+'.mp4'
        answer=row['ground_truth']
        
        if type(candidates)== np.ndarray:
            mc_cnt+=1
        elif candidates is None:
            oe_cnt+=1
            continue
        else:
            print(type(candidates))
        
        data_list_info.append({
            "task_name": task_name+question_type,
            "video_name": video_path,
            "question_id": question_id,
            "question": question,
            "answer_number":choices.index(answer),
            "candidates": candidates.tolist(),
            "answer": answer,
        })
            
    print(oe_cnt)
    print(mc_cnt)

    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/MMVU/val_qa.json", "w",encoding='utf-8') as f:
        json.dump(data_list_info, f, indent=4)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
