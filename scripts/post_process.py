#给定指定视频的路径和帧数，在该视频中抽取指定帧数，拼成一张大图输出到指定位置
from decord import VideoReader, cpu
import numpy as np
from PIL import Image

video_path="/mnt/cloud_disk/public_data/VideoMME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/data/9jjTGpWmc5U.mp4"
frames=32

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

def get_rows_cols(frames):
    factors = []
    for i in range(1, int(np.sqrt(frames)) + 1):
        if 32 % i == 0:
            factors.append((i, 32 // i))
    # 计算每个因数对的比例
    ratios = [max(f) / min(f) for f in factors]
    # 找到比例最接近 1 的因数对，即为最接近正方形的布局
    rows, cols = factors[np.argmin(ratios)]
    
    return rows,cols

def cat_imgs(images,frames,output_path):
    
    rows,cols=get_rows_cols(frames)
    
    # 初始化一个空白的大图
    height = rows * images.shape[1]
    width = cols * images.shape[2]
    big_image = np.zeros((height, width, 3), dtype=images.dtype)

    # 将小图依次放入大图的相应位置
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            start_y = i * images.shape[1]
            end_y = start_y + images.shape[1]
            start_x = j * images.shape[2]
            end_x = start_x + images.shape[2]
            big_image[start_y:end_y, start_x:end_x] = images[index]

    # 将 numpy.ndimagesay 转换为 PIL 图像
    img = Image.fromarray(big_image)
    # 保存图像
    img.save(output_path)
    

def post_process(video_path,frames,output_path):
    imgs=load_video(video_path,frames)
    # print(imgs.shape)
    # print(type(imgs))
    cat_imgs(imgs,frames,output_path)
    print(f'already saved to {output_path}')
    
