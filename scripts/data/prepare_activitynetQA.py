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

task_name=" AcivityNet "
data_dir="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/ActivityNetQA/videos/all_test/"
choices=["(A)","(B)","(C)","(D)","(E)"]
choices_=["A.","B.","C.","D.","E."]

def main():
    data_list_info = []
    
    df = pd.read_parquet('/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/ActivityNetQA/data/test-00000-of-00001.parquet')
    
    fake_path=0
    true_path=0
    for index, row in df.iterrows():
        video_path=data_dir+'v_'+row['video_name']+'.mp4'
        path = Path(video_path)     
        if not path.exists():
            fake_path+=1
            continue
        
        if row['answer'] in ['yes','no']:
            continue

        # data_list_info.append({
        #         "task_name": task_name,
        #         "video_name": video_path,
        #         "question_id": row['question_id'],
        #         "question": row['question']+'?',
        #         "answer_number": 0 if row['answer']=='yes' else 1,
        #         "candidates": ["A. yes","B. no"],
        #         "answer": row['answer'],
        #     })
        data_list_info.append({
                "task_name": task_name,
                "video_name": video_path,
                "question_id": row['question_id'],
                "question": row['question']+'?',
                "answer_number": -1,
                "candidates": -1,
                "answer": row['answer'],
            })
        true_path+=1
        
    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/ActivityNetQA/val_qa_others.json", "w") as f:
        json.dump(data_list_info, f, indent=4)

    # print(f'fake path :{fake_path}')
    print(f'true path :{true_path}')
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
