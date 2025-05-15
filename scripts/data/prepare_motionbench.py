#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json

task_name=" MotionBench "
data_dir="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/MotionBench/MotionBench/all-videos/"
qa_file="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/MotionBench/MotionBench/video_info.meta.jsonl"

choices=["A","B","C","D","E"]

def main():
    data_list_info = []
    
    with open(qa_file, "r") as file:
        for line in file:
            data = json.loads(line)
            if data['qa'][0]['answer'] == 'NA':
                continue
            
            question_candidates=data['qa'][0]['question']
            question_candidates=question_candidates.split('\n')
            question=question_candidates[0]
            if question_candidates[-1]=="":
                question_candidates.pop()
            candidates=question_candidates[1:]
            
            data_list_info.append({
                "task_name": task_name+data['question_type'],
                "video_name": data_dir+data['video_path'],
                "question_id": 0,
                "question": question,
                "answer_number": choices.index(data['qa'][0]['answer']),
                "candidates": candidates,
                "answer": data['qa'][0]['answer'],
            })

    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/MotionBench/val_qa.json", "w") as f:
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
