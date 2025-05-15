#对egoschema中的question做预处理，筛选出需要complex reasoning的问题

import os
import argparse
import csv
import pandas as pd
import json
from tools.evaluate_question import categories,classify_question
from pathlib import Path

gt_path="total.json"
task_name=" VideoMMMU "

all_categories=['fact_retrieval', 'process_description', \
    'causal_reasoning', 'theme_summary', 'comparative_analysis', \
        'behavior_inference', 'tool_analysis', 'key_moment', \
        'efficiency_evaluation', 'interaction_analysis','other']

categories_need_reasoning=["causal_reasoning","theme_summary","behavior_inference"]


def main():
    all_data_dict={}
    for category in all_categories:
        all_data_dict[f'{category}_data']=[]
    reasoning_samples=0
    
    with open(gt_path,'r',encoding='utf-8') as file:
        data=json.load(file)
    
    print(len(data))
    
    for row in data:
        question_id=row['question_id']
        question=row['question']
        video_name=row['video_name']
        candidates=row['candidates']
        answer_number=row['answer_number']
        category=classify_question(question)
        video_path=video_name

        row['video_name']=video_path
        if category in categories_need_reasoning:
            need_reasoning=True
            reasoning_samples+=1
        else:
            need_reasoning=False
        row['need_reasoning']=need_reasoning
        
        all_data_dict[f'{category}_data'].append({
            "task_name": task_name,
            "video_name": video_path,
            "question_id": question_id,
            "question": question,
            "answer_number": answer_number,
            "candidates": candidates,
            "answer": candidates[int(answer_number)],
            "category":category,
            "need_reasoning":need_reasoning
        })
    print(reasoning_samples)
    
    for cat in all_categories:
        with open(f"{cat}_filter_val_qa.json", "w") as f:
            json.dump( all_data_dict[f'{cat}_data'], f, indent=4)
        

if __name__ == "__main__":
    main()
