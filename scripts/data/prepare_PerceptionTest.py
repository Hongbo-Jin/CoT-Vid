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

task_name=" PerceptionTest "

test_data_dir="/mnt/cloud_disk/public_data/PerceptionTest/videos/"
val_data_dir="/mnt/cloud_disk/public_data/PerceptionTest_val/videos/"

test_data_path="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/PerceptionTest/all_test.json"

test_gt="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/PerceptionTest/mc_question/test-00000-of-00001.parquet"
val_gt="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/PerceptionTest/mc_question_val/validation-00000-of-00001.parquet"


choices=["(A)","(B)","(C)","(D)","(E)"]
choices_=["A. ","B. ","C. ","D. ","E. ","F. "]


def main():
    
    test_gt_data=pd.read_parquet(test_gt)
    val_gt_data=pd.read_parquet(val_gt)
    data_list_info = []
    fake_path=0
    true_path=0
    for index,val_sample in val_gt_data.iterrows():
        video_path=val_data_dir+val_sample['video_name']+'.mp4'
        
        path=Path(video_path)
        if not path.exists():
            fake_path+=1
            # print(video_path)
        else:
            true_path+=1

        question=val_sample['question']
        options=list(val_sample['options'])
        for idx in range(len(options)):
            options[idx]=choices_[idx]+options[idx]
        
        answer_id=int(val_sample['answer_id'])
        data_list_info.append({
                "task_name": task_name+val_sample['area']+val_sample['reasoning'],
                "video_name": video_path,
                "question_id": val_sample['video_name']+'_'+str(val_sample['question_id']),
                "question": question,
                "answer_number":answer_id,
                "candidates": options,
                "answer": options[answer_id],
            })

    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/PerceptionTest/val_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)
    print(fake_path)
    print(true_path)
    
    val_sample=-1
    
    fake_path=0
    true_path=0
    for index,test_sample in test_gt_data.iterrows():
        video_path=test_data_dir+test_sample['video_name']+'.mp4'
        
        path=Path(video_path)
        if not path.exists():
            fake_path+=1
        else:
            true_path+=1

        question=test_sample['question']
        options=list(test_sample['options'])
        for idx in range(len(options)):
            options[idx]=choices_[idx]+options[idx]
        
        # answer_id=int(test_sample['answer_id'])
        data_list_info.append({
                "task_name": task_name,
                "video_name": video_path,
                "question_id": test_sample['video_name']+'_'+str(test_sample['question_id']),
                "question": question,
                # "answer_number":answer_id,
                "candidates": options,
                # "answer": options[answer_id],
            })

    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/PerceptionTest/test_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)
    print(fake_path)
    print(true_path)

if __name__ == "__main__":
    main()
