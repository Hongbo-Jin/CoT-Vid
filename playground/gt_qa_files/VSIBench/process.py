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
import numpy as np

task_name=" VSIBench "
data_dir="/mnt/cloud_disk/public_data/VSIBench/"
qa_file="/mnt/cloud_disk/public_data/VSIBench/test-00000-of-00001.parquet"

choices=["A","B","C","D","E"]

def main():
    data_list_info = []
    
    oe_cnt=0
    mc_cnt=0
    err_path=0
    
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

    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VSIBench/val_qa.json", "w",encoding='utf-8') as f:
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
