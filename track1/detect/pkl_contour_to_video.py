import os
import sys
import time
import random
import math
import numpy as np
#import skimage.io
import imageio
import cv2
import tqdm
import pickle

import visualize

def main():
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
    
    # video directory
    video_dir = "../../../../aic2018/track1/track1_videos/"
    detect_dir = "../../../../aic2018/track1/detect/"
    save_dir = "../../../../aic2018/track1/detect_videos/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # set upper and lower bound of visualizd frames
    ub = 1800
    lb = 0
    
    videonames = [x for x in os.listdir(video_dir) if x.startswith("Loc4")]
    print(videonames)
    for videoname in videonames:
        print("Processing video {}...".format(videoname))
        # read video
        video_file = os.path.join(video_dir, videoname)
        vid = imageio.get_reader(video_file,  'ffmpeg')
        # load pkl files
        pkl_dir = os.path.join(detect_dir, videoname)
        pkl_files = [str(x).zfill(7)+".pkl" for x in range(lb, ub+1)]#sorted([x for x in os.listdir(pkl_dir)])
        # write output video
        fps = vid.get_meta_data()['fps']
        writer = imageio.get_writer(os.path.join(save_dir, videoname.replace(".mp4", "_detect.mp4")), fps=fps)
        
        ub = min(ub, vid.get_length()-1)
        lb = max(lb, 0)
        pbar = tqdm.tqdm(total = ub-lb+1)
        for pkl in pkl_files:
            fnum = int(pkl.replace(".pkl", ""))
            if fnum<lb:
                continue
            elif fnum > ub:
                break
            r = pickle.load(open(os.path.join(pkl_dir,pkl), "rb"))
            mask_image = visualize.draw_contours(vid.get_data(fnum), r['rois'], r['contours'], 
                                                 r['class_ids'],class_names, r['scores'])
            cv2.putText(mask_image, str(fnum), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
            writer.append_data(mask_image)
            pbar.update(1)
        pbar.close()
        writer.close()
            
            
if __name__ == "__main__":
    main()
