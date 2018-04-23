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

#import visualize

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
    video_dir = "../../../../detrac/data/Insight-MVT_Annotation_Train"
    detect_dir = "../../../../detrac/data/train_detect_output"
    save_dir = "../../../../detrac/data/train_detect_txt"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # set upper and lower bound of visualizd frames
    lb_init = 0
    ub_init = 18000
    assert ub_init>lb_init, "upper bound < lower bound"

    videonames = [x for x in os.listdir(video_dir) if x.startswith("MVI")]
    for videoname in videonames:
        print("Processing video {}...".format(videoname))
        # load pkl files
        pkl_dir = os.path.join(detect_dir, videoname)
        pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")])
        f = open(os.path.join(save_dir, videoname.split(".")[0]+"_det.txt"), "w+")
        ub = min(ub_init, len(pkl_files)-1)
        lb = max(lb_init, 0) 
        pbar = tqdm.tqdm(total = ub-lb+1)
        for pkl in pkl_files:
            fnum = int(pkl.replace(".pkl", "").replace('img', ''))
            if fnum<lb:
                continue
            elif fnum > ub:
                break
            r = pickle.load(open(os.path.join(pkl_dir,pkl), "rb"))
            #assert isinstance(r, dict), (r[0], pkl) 
            if isinstance(r, tuple):
                 r = r[1] # old data format: (frame number, r)
            for i, roi in enumerate(r['rois']):
                y1, x1, y2, x2 = roi
                conf = r['scores'][i]
                cid = class_names[r['class_ids'][i]]
                fid = fnum
                if cid in ['car', 'truck', 'bus']:
                    det = [fid, -1, x1, y1, abs(x2-x1), abs(y2-y1), conf, -1, -1, -1]
                    string = ", ".join([str(x) for x in det])+"\n"
                    f.write(string)
            pbar.update(1)
        f.close()  
        pbar.close()
            
            
if __name__ == "__main__":
    main()
