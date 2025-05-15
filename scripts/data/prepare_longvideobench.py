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

task_name=" LongVideoBench "
data_dir="/mnt/cloud_disk/public_data/LongVideoBench/videos/"

val_data_path="/mnt/cloud_disk/public_data/LongVideoBench/lvb_val.json"
test_data_path="/mnt/cloud_disk/public_data/LongVideoBench/lvb_test_wo_gt.json"


choices=["(A)","(B)","(C)","(D)","(E)"]
choices_=["A. ","B. ","C. ","D. ","E. ","F. "]


def main():
    data_list_info = []
    df = pd.read_json(val_data_path)
    
    for index, row in df.iterrows():
        video_path=data_dir+row['video_path']
        
        candidates=row['candidates']
        for idx in range(len(candidates)):
            candidates[idx]=choices_[idx]+candidates[idx]
        data_list_info.append({
                "task_name": task_name+row['topic_category'],
                "video_name": video_path,
                "question_id": row['id'],
                "question": row['question'],
                "answer_number": row['correct_choice'],
                "candidates": candidates,
                "answer": candidates[row['correct_choice']],
            })

    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/LongVideoBench/val_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)


if __name__ == "__main__":
    main()
