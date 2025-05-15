#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import argparse
import csv
import json

import pandas as pd

mvbench_path="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/MVBench"
data_list = {
    "Action Sequence": ("action_sequence.json", f"{mvbench_path}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", f"{mvbench_path}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", f"{mvbench_path}/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", f"{mvbench_path}/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", f"{mvbench_path}/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", f"{mvbench_path}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", f"{mvbench_path}/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", f"{mvbench_path}/video/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", f"{mvbench_path}/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", f"{mvbench_path}/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", f"{mvbench_path}/video/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", f"{mvbench_path}/video/nturgbd/", "video", False),
    "Character Order": ("character_order.json", f"{mvbench_path}/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", f"{mvbench_path}/video/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", f"{mvbench_path}/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
}

sub_tasks=data_list.keys()

options=['A','B','C','D','E']

def main(args, task_name="MVBench"):
    data_list_info = []
    
    for sub_task in sub_tasks:
        s0,s1,s2,s3=data_list[sub_task]
        if not s3:
            pass
        if not s2=='video':
            continue
        sub_task_path=args.qa_file+s0
        try:
            with open(sub_task_path, 'r') as file:
                sub_task_data = json.load(file)
        except Exception as e:
            print(f"发生错误: {e}")

            
        for sample in sub_task_data:
          
            data_list_info.append({
                "task_name" : "MVBench "+sub_task,
                "video_name": s1+sample['video'],
                "question_id" :0,
                "question": sample['question'],
                "answer_number": int(sample['candidates'].index(sample['answer'])),
                "candidates": sample['candidates'],
                "answer": sample['answer'],
            })
    
    folder = f"playground/gt_qa_files/{task_name}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/val_qa.json", "w") as f:
        json.dump(data_list_info, f, indent=4)
        

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", help=" ", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
