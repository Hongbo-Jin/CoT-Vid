#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json
import pandas as pd
from pathlib import Path

task_name=" nextqa "
data_dir="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/nextqa/"
choices=["(A)","(B)","(C)","(D)","(E)"]
choices_=["A.","B.","C.","D.","E."]
# qa_file="/mnt/cloud_disk/public_data/NExTQA/nextqa_mc.parquet"
qa_file="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/nextqa/test-00000-of-00001.parquet"

def main():
    data_list_info = []
    df=pd.read_parquet(qa_file)

    for index,row in df.iterrows():
    
        options=[
            "A. "+row['a0'],
            "B. "+row['a1'],
            "C. "+row['a2'],
            "D. "+row['a3'],
            "E. "+row['a4'],
        ]
        video_path=data_dir+str(row['video'])+'.mp4'
        
        data_list_info.append({
            "task_name": task_name+row['type'],
            "video_name": video_path,
            "question_id": -1,
            "question": row['question'],
            "answer_number": row['answer'],
            "candidates": options,
            "answer": options[row['answer']],
        })
        
    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/nextqa/total_val_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)

if __name__ == "__main__":
    main()
