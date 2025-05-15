#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json
from pathlib import Path
import pandas as pd

task_name=" LVBench "
data_dir="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/LVBench/all_videos/"

choices=["(A)","(B)","(C)","(D)","(E)"]
choices_=["A","B","C","D","E"]


def main():
    data_list_info = []
    df = pd.read_json('/mnt/cloud_disk/public_data/LVBench/video_info.meta.jsonl', lines=True)    
    
    fake_path=0
    true_path=0
    for index, row in df.iterrows():
        
        video_path=data_dir+row['key']+'.mp4'
        path = Path(video_path)     
        if not path.exists():
            fake_path+=1
            continue
        qa=row['qa']

        for qa_sample in qa:
            question_candidates=qa_sample['question'].split('\n')
            question=question_candidates[0]
            candidates=question_candidates[1:]
            question_type=qa_sample['question_type'][0]
            answer=qa_sample['answer']

            for idx in range(len(candidates)):
                candidates[idx]=candidates[idx].replace('(A)',"A. ")
                candidates[idx]=candidates[idx].replace('(B)',"B. ")
                candidates[idx]=candidates[idx].replace('(C)',"C. ")
                candidates[idx]=candidates[idx].replace('(D)',"D. ")

            true_path+=1
            data_list_info.append({
                    "task_name": task_name+row['type']+' '+question_type,
                    "video_name": video_path,
                    "question_id": qa_sample['uid'],
                    "question": question,
                    "answer_number": choices_.index(answer),
                    "candidates": candidates,
                    "answer": answer,
                })

    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/LVBench/val_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)
    
    print(f"fake path:{fake_path}")
    print(f"true path:{true_path}")
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", help="Path to EgoSchema.csv", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
