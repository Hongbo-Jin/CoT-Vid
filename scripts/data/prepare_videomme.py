
import os
import argparse
import csv
import json
import pandas as pd

task_name="VideoMME"
data_dir="/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/data/videomme/"
qa_file="/mnt/cloud_disk/public_data/VideoMME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"

choices=['A','B','C','D','E']

def main():
    short_data_list_info = []
    medium_data_list_info= []
    long_data_list_info= []
    
    df = pd.read_parquet('/mnt/cloud_disk/public_data/VideoMME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet')
    print(df.head())
    print(df.columns)
    print(len(df))
    
    fake_path=0
    
    for index, row in df.iterrows():
        video_id, duration, domain, sub_category, url, videoID, \
        question_id, task_type, question, options, answer=row['video_id'],row['duration'],row['domain'], \
           row['sub_category'],row['url'],row['videoID'],row['question_id'],row['task_type'], \
            row['question'],row['options'].tolist(),row['answer']
       
        video_path='/mnt/cloud_disk/public_data/VideoMME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/data/'+videoID+'.mp4'
        
        if duration=='short':
            short_data_list_info.append({
                "task_name": task_name+'-'+task_type+'-'+domain+'-'+sub_category,
                "video_name": video_path,
                "question_id": 0,
                "question": question,
                "answer_number": choices.index(answer),
                "candidates": options,
                "answer": answer,
            })
        elif duration=='long':
            long_data_list_info.append({
                "task_name": task_name+'-'+task_type+'-'+domain+'-'+sub_category,
                "video_name": video_path,
                "question_id": 0,
                "question": question,
                "answer_number": choices.index(answer),
                "candidates": options,
                "answer": answer,
            })
        else:
            medium_data_list_info.append({
                "task_name": task_name+'-'+task_type+'-'+domain+'-'+sub_category,
                "video_name": video_path,
                "question_id": 0,
                "question": question,
                "answer_number": choices.index(answer),
                "candidates": options,
                "answer": answer,
            })
            
    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VideoMME/short_val_qa.json", "w") as f:
        json.dump(short_data_list_info, f, indent=4)
    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VideoMME/medium_val_qa.json", "w") as f:
        json.dump(medium_data_list_info, f, indent=4)
    with open("/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VideoMME/long_val_qa.json", "w") as f:
        json.dump(long_data_list_info, f, indent=4)
    #     if not os.path.exists(video_path):
    #         print(f"路径 {video_path} 不存在。")
    #         fake_path+=1

if __name__ == "__main__":
    main()
