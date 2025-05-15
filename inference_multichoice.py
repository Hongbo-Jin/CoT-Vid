from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

import argparse
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from llava.utils import disable_torch_init
import math

from difflib import SequenceMatcher
from collections import Counter

import spacy
from sklearn.metrics.pairwise import cosine_similarity
import shortuuid

tasks_need_options=["nextqa","MVBench","egoschema","mlvu",]
tasks_wo_path=[]
tasks_have_path=["egoschema","nextqa","MVBench","VideoEspresso",\
    "MotionBench","Activity","LVBench","LongVideoBench",\
        "PerceptionTestVal","videomme","TempCompass",\
            "VideoMMMU","VSIBench"]

options=['A','B','C','D','E']

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

def llava_inference(
    video_frames,
    question,
    candidates,
    model,
    tokenizer,
    image_processor,
    sample=None,
    conv_template = "qwen_1_5",
    device="cuda",
):

    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)
    
  # Prepare conversation input
    question = f"{DEFAULT_IMAGE_TOKEN}\nThe input consists of a sequence of key frames from a video.Please answer the following questions:\n"+question
    
    if args.task_name in tasks_need_options:
        options=['A','B','C','D','E']
        for option,candidate in zip(options,candidates):
            question+="\n"+option+'.'+candidate
    else:
        for candidate in candidates:
            question+="\n"+candidate
    question+="\nOnly select the best answer."
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # print(prompt_question)
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    
    # print(text_outputs)
    # exit(0)
    return text_outputs,None

def Summarize(
    model,
    tokenizer,
    question,
    device,
    image_sizes,
    image_tensors,
    with_question=True,
    temperature=0,
):
    if with_question:
        summary_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nThe input consists of a sequence of key frames from a video.\nSummarize the main content in the video, paying special attention to content related to the question:"+question+".\nContent unrelated to the question can be summarized more briefly.\n<|im_end|>\n<|im_start|>assistant\n"
    else:
        summary_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nThe input consists of a sequence of key frames from a video.\nSummarize the main content in the video.\n<|im_end|>\n<|im_start|>assistant\n"
    # print(summary_prompt)
    summary_input_ids = tokenizer_image_token(summary_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    summary_cont = model.generate(
            summary_input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=4096,
            modalities=["video"],
        )
    summary_text_outputs = tokenizer.batch_decode(summary_cont, skip_special_tokens=True)[0]
    
    return summary_text_outputs


def standard_cot(
    model,
    tokenizer,
    question,
    device,
    image_sizes,
    image_tensors,
    candidates,
    summary_text_outputs,
    with_summary=False,
    sample_set=None,
):  
    # prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>"+ \
    #     "\n<|im_start|>user\n<image>\nThe input consists of a sequence of key frames from a video.\n" + \
    #     "The main content of the video is summarized below:\n"+summary_text_outputs+ \
    #     "\nHere is a question about the video:\n"+question+".\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"
    prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nThe input consists of a sequence of key frames from a video.Here is a question about the video:\n"+question+".\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    sample_set['prompt_1']=prompt
    sample_set['output_1']=text_outputs
    if with_summary:
        response_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nThe input consists of a sequence of key frames from a video."+ \
            "The main content of the video is summarized below:\n"+summary_text_outputs+ \
            "\nHere is a question about the video:\n"+question
    else:
        response_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nThe input consists of a sequence of key frames from a video."+ \
            "\nHere is a question about the video:\n"+question
    if args.task_name in tasks_need_options:
        for option,candidate in zip(options,candidates):
            response_prompt+="\n"+option+'.'+candidate
    else:
        for candidate in candidates:
            response_prompt+="\n"+candidate
    
    response_prompt+=".\nLet's think step by step:\n"+ \
      text_outputs+"\nOnly select the best answer.So the answer is:\n<|im_end|>\n<|im_start|>assistant\n"
    sample_set['prompt2']=response_prompt
    input_ids = tokenizer_image_token(response_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    sample_set['output_2']=text_outputs
    return text_outputs,None


def self_verify(
    image_tensors,
    image_sizes,
    question,
    candidates,
    model,
    tokenizer,
    answer,
    device,
):
    rethink_choices=[
        "A. Very reliable","B. generally reliable",
        "C. not very reliable","D. absolutely impossible" ]
    
    answer_choice=answer[0]
    choices_idx=['A','B','C','D','E']
    answer_idx=choices_idx.index(answer_choice)
    answer=candidates[answer_idx]
    
    proposed_answer=answer
    
    if args.task_name in tasks_need_options:
        for option,candidate in zip(options,candidates):
            question+="\n"+option+'.'+candidate
    else:
        for candidate in candidates:
            question+="\n"+candidate
    
    question+="\nHere is an answer to this question:\n"+answer + \
        "\nHow reliable do you think this answer is?\n"+ \
        "A. Very reliable \nB. generally reliable \nC. not very reliable \nD. absolutely impossible\n" + \
        "Only select the best answer."
    
    question_ = f"{DEFAULT_IMAGE_TOKEN}\n"+ \
    "The input consists of a sequence of key frames from a video."+ \
    "\nPlease answer the following questions:\n"+question
    
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], question_)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # print(prompt_question)
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    verify_results = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    
    if rethink_choices[0] == 'A':
        verify_results= rethink_choices[0]
    elif rethink_choices[0]=='B':
        verify_results= rethink_choices[1]
    elif rethink_choices[0]=='C':
        verify_results= rethink_choices[2]
    else :
        verify_results= rethink_choices[3]
    # print(proposed_answer)
    # print(verify_results)
    return proposed_answer,verify_results

def rethink_step(
    image_tensors,
    image_sizes,
    question,
    candidates,
    model,
    tokenizer,
    device,
    proposed_answer,
    verify_results,
    with_summary=False,
    summary_text_outputs=None,
):
    if args.task_name in tasks_need_options:
        for option,candidate in zip(options,candidates):
            question+="\n"+option+'.'+candidate
    else:
        for candidate in candidates:
            question+="\n"+candidate
    question+="\nOnly select the best answer.\n"
    ori_question=question
    
    question+="The best answer is :"+proposed_answer
    question+="\nLet us rethink about the answer.How reliable do you think this answer is?" + \
        "\nA. Very reliable \nB. generally reliable \nC. not very reliable \nD. absolutely impossible\n" + \
        "I think :"+verify_results+"\nGiven that,please answer the following question again:\n"+ori_question
    
    if with_summary:
        question_ = f"{DEFAULT_IMAGE_TOKEN}\n"+ \
        "The input consists of a sequence of key frames from a video.The main content of the video is summarized below:\n"+ \
        summary_text_outputs+ \
        "\nPlease answer the following questions:\n"+question
    else:
        question_ = f"{DEFAULT_IMAGE_TOKEN}\n"+ \
        "The input consists of a sequence of key frames from a video."+ \
        "\nPlease answer the following questions:\n"+question
    
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], question_)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    # print(prompt_question)

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    # print(text_outputs)
    return text_outputs,prompt_question

def llava_inference_cot(
    video_frames,
    question,
    candidates,
    model,
    tokenizer,
    image_processor,
    sample=None,
    vote_num=5,
    sample_set=None,
):
    device="cuda"
    conv_template = "qwen_1_5"
    
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)
    image_sizes = [frame.size for frame in video_frames]
    
    summary_text_outputs=Summarize(
        model,
        tokenizer,
        question,
        device,
        image_sizes,
        image_tensors,
        with_question=True,
    )
    sample_set['summary']=summary_text_outputs
    # print(summary_text_outputs)
    return standard_cot(
        model,
        tokenizer,
        question,
        device,
        image_sizes,
        image_tensors,
        candidates,
        summary_text_outputs=summary_text_outputs,
        with_summary=True,
        sample_set=sample_set
    )
    
    ori_question=question
    if args.task_name in tasks_need_options:
        for option,candidate in zip(options,candidates):
            question+="\n"+option+'.'+candidate
    else:
        for candidate in candidates:
            question+="\n"+candidate
    question+="\nOnly select the best answer.\n"
    
    question_ = f"{DEFAULT_IMAGE_TOKEN}\n"+ \
    "The input consists of a sequence of key frames from a video.The main content of the video is summarized below:\n"+ \
    summary_text_outputs+ \
    "\nPlease answer the following questions:\n"+question
    
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question_)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # print(prompt_question)
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    
    return text_outputs,None

def llava_inference_cot_sc(
    video_frames,
    question,
    candidates,
    model,
    tokenizer,
    image_processor,
    sample=None,
    vote_num=3,
    sample_set=None,
):
    device="cuda"
    conv_template = "qwen_1_5"
    
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)
    image_sizes = [frame.size for frame in video_frames]

    summary_texts=[]
    for idx in range(vote_num):
        summary_text_output=Summarize(
        model,
        tokenizer,
        question,
        device,
        image_sizes,
        image_tensors,
        with_question=True,
        temperature=args.temperature)
        
        summary_texts.append(summary_text_output)
    
    clusters=[]
    for sum_text in summary_texts:
            added_to_cluster = False
            for cluster in clusters:
                if SequenceMatcher(None, sum_text, cluster[0]).ratio() >= args.similarity_threshold:
                    cluster.append(sum_text)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([sum_text])
    freq=0
    for cluster in clusters:
        if freq<len(cluster):
            freq=len(cluster)
            max_summary_outputs=cluster[0]
    
    sample_set['summary']=max_summary_outputs
    
    question_ = f"{DEFAULT_IMAGE_TOKEN}\n"+ \
    "The input consists of a sequence of key frames from a video.\nThe main content of the video is summarized below:\n" + \
        max_summary_outputs + \
        "\nPlease answer the following questions:\n"+question
    if args.task_name in tasks_need_options:
        for option,candidate in zip(options,candidates):
            question_+="\n"+option+'.'+candidate
    else:
        for candidate in candidates:
            question_+="\n"+candidate
    question_+="\nOnly select the best answer."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question_)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    sample_set['prompt']=prompt_question
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    
    return response,None


def evaluate_candidates(
    candidates,
    question,
    score_model
):
    scores=[]
    for candidate in candidates:
        features = {}
        # 文本预处理
        q_doc = score_model(question)
        s_doc = score_model(candidate)
    
        # 特征1：语义相关性（双向相似度）
        q_vector = q_doc.vector.reshape(1, -1)
        s_vector = s_doc.vector.reshape(1, -1)
        features['semantic_sim'] = (cosine_similarity(q_vector, s_vector)[0][0] + 1) / 2  # 归一化到0-1
        
        scores.append(features['semantic_sim'])
        
    return scores

def filter_best_of_n(
  candidates,
  question,
  score_model
):
    assert type(candidates)==list
    total_num=len(candidates)
    scores=evaluate_candidates(candidates,question,score_model)
    
    best_score=-1
    best_answer=""
    for idx in range(total_num):
        if scores[idx]>best_score:
            best_score=scores[idx]
            best_answer=candidates[idx]
    return best_answer

def llava_inference_BoN(
    video_frames,
    question,
    candidates,
    model,
    tokenizer,
    image_processor,
    sample=None,
    vote_num=9,
    score_model=None
):
    device="cuda"
    conv_template = "qwen_1_5"
    
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)
    image_sizes = [frame.size for frame in video_frames]

    summary_texts=[]
    for idx in range(vote_num):
        summary_text_output=Summarize(
        model,
        tokenizer,
        question,
        device,
        image_sizes,
        image_tensors,
        with_question=True,
        temperature=args.temperature)
        
        summary_texts.append(summary_text_output)
        
    best_summary=filter_best_of_n(summary_texts,question=question,score_model=score_model)
    
    question_ = f"{DEFAULT_IMAGE_TOKEN}\n"+ \
    "The input consists of a sequence of key frames from a video.\nThe main content of the video is summarized below:\n" + \
        best_summary + \
        "\nPlease answer the following questions:\n"+question
    if args.task_name in tasks_need_options:
        for option,candidate in zip(options,candidates):
            question_+="\n"+option+'.'+candidate
    else:
        for candidate in candidates:
            question_+="\n"+candidate
    question_+="\nOnly select the best answer."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question_)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

    return response,None

def run_inference(args):
    error_video_paths=[]
    disable_torch_init()

    pretrained = args.model_path
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
    }
    #only used with best of N
    score_model = spacy.load("en_core_web_sm")
    
    tokenizer, model, image_processor, max_length = \
    load_pretrained_model(pretrained, None, model_name, device_map=device_map, \
                            load_8bit=args.bits==8, load_4bit=args.bits==4, \
                            attn_implementation="sdpa", **llava_model_args)

    
    model.eval()

    # Load questions and answers
    gt_qa_pairs = json.load(open(args.gt_file, "r"))
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(
        os.path.join(args.output_dir, f"{args.output_name}.json"), "w")

    # Iterate over each sample in the ground truth file
    for index in tqdm(range(0, len(gt_qa_pairs), args.sample_rate)):
        sample=gt_qa_pairs[index]
        
        question = sample["question"]
        answer = sample["answer"]
        candidates = sample["candidates"]
        if args.task_name=="mlvu":
            video_name = sample["video"]
            answer_number=candidates.index(answer)
        else:
            video_name = sample["video_name"]
            answer_number = sample["answer_number"]

        # Load video
        if args.task_name in tasks_have_path:
            video_path=sample["video_name"]
        else:
            video_path = os.path.join(args.video_dir, video_name)
        
        # 定义路径
        v_path = Path(video_path)

        # 判断路径是否存在
        if not v_path.exists():
            print(f"路径 '{v_path}' 不存在")
            continue
        
        try:
            video_frames = load_video(video_path, args.num_frames)
        except Exception as e:
            # print(f"Error loading video {video_path}: {e}")
            error_video_paths.append((video_path,e))
            continue
        # print(video_frames.shape) # (16, 1024, 576, 3)

        sample_set = {
            "video_path":video_path,
            "question": question,
            "answer_number": answer_number,
            "candidates": candidates,
            "answer": answer,
        }
        
        # if "need_reasoning" in sample.keys() and sample['need_reasoning']==False:
        #     inference_func=llava_inference
        if args.inference_mode=="base":
            inference_func=llava_inference
        elif args.inference_mode=="cot":
            inference_func=llava_inference_cot
        elif args.inference_mode=="cot_sc":
            inference_func=llava_inference_cot_sc
        elif args.inference_mode=="BoN":
            inference_func=llava_inference_BoN
        else:
            print('error')
            exit(0)
            
        if args.inference_mode=="BoN":
            output,reasoninig_chain = inference_func(
            video_frames,
            question,
            candidates,
            model,
            tokenizer,
            image_processor,
            sample,
            score_model=score_model
        )
        else:   
            # Run inference on the video
            output,reasoninig_chain = inference_func(
                video_frames,
                question,
                candidates,
                model,
                tokenizer,
                image_processor,
                sample,
                sample_set=sample_set
            )
        
        sample_set["pred"] = output
        if output[0] not in ['A','B','C','D','E']:
            print(output)
            
        if args.output_reasoning_chain:
            sample_set['reasoninig_chain']=reasoninig_chain
        if args.output_category:
            sample_set['category']=sample['category']
        
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
    
    with open(f'/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/outputs/error_path/{args.task_name}_error.txt', 'w') as file:
        for item in error_video_paths:
            file.write(f"{item}\n")  # 每个元素占一行

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--rope_scaling_factor", type=int, default=1)
    parser.add_argument("--inference_mode", type=str, default="base",choices=["base","cot","cot_sc","BoN"],required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--task_name", type=str, default='')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--similarity_threshold", type=float, default=0.8)
    parser.add_argument("--sample_rate",type=int, default=1)
    parser.add_argument("--rethink",type=bool, default=False)
    parser.add_argument("--output_reasoning_chain",type=bool, default=False)
    parser.add_argument("--output_category",type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
