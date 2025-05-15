#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json

task_name=" VideoEspresso "
data_dir="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/videoespresso/"
choices=["(A)","(B)","(C)","(D)","(E)"]
choices_=["A.","B.","C.","D.","E."]
qa_file="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/videoespresso/bench_hard.json"

def main():
    data_list_info = []
    
    with open(qa_file, 'r') as file:
        data = json.load(file)
        for idx in range(len(data)):
            sample=data[idx]
            core_frame_path,core_frame_captions,question, \
            answer,key_item,evidence,task,\
            answer_option,video_path,options=sample.values()
            
            for idx,option in enumerate(options):
                options[idx]=choices_[idx]+options[idx][4:]
                
            data_list_info.append({
                "task_name": task_name+task,
                "video_name": data_dir+video_path,
                "question_id": 0,
                "question": question,
                "answer_number": choices.index(answer_option),
                "candidates": options,
                "answer": answer,
            })

    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VideoEspresso/val_qa.json", "w") as f:
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
