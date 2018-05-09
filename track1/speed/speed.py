import tensorflow as tf
import numpy as np

import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
import imageio
import cv2
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.io as sio

frame_w, frame_h = 1920, 1080

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=x#np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def smoothv(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def coord2dto3d(m, pts):
    """
    m: [3,3]
    pts: [n, 3] or [n, 2]
    return [n, 2]
    """
    if pts.shape[1] == 2:
        pts = np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1)
    X = pts # (u,v,1)
    N = 2
    U1 = np.dot(X, m) # transform in homogeneous coordinates
    UN = np.tile(U1[:, -1][:,np.newaxis], (1, N)) # replicate the last column of U
    U = U1[:, :-1] / UN
    return U


def extract_bottom(contour):
    """
    Think!!! how to improve code by removing for loop?
    """
    bottom_contour = []
    for c in contour:
        c = np.squeeze(c)
        if len(c.shape) == 1:
            c = np.expand_dims(c, axis=0)
        c = list(c)
        interp_c = []
        for i, _ in enumerate(c[:-1]):
            interp_list = [[p, c[i][1]] for p in range(c[i][0]+1, c[i+1][0])]
            interp_c.extend(c[i:i+1] + interp_list)
        interp_c.append(c[-1])
        c = np.array(interp_c).astype("int32")
        c = c[np.lexsort((c[:,1],c[:,0]))]
        newc = {k: v for k, v in c}
        newc = np.asarray(list(newc.items()))
        newc = np.expand_dims(newc, axis=1)
        bottom_contour.append(newc)
    return bottom_contour


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
    video_dir = "../data/track1_videos/"
    detect_dir = "../data/detect_output_pkl/"
    track_dir = "../data/track_output/"
    save_dir = "../output/speed_videos"
    savetxt_dir = "../output/speed_txt"
    mat_file = "./track1_M.mat"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(savetxt_dir):
        os.makedirs(savetxt_dir)

    # read transform matrix
    M = sio.loadmat(mat_file)
    """
   -0.0475   -0.0097   -0.0001
   -0.0765   -0.7754   -0.0043
   55.3943  291.6660    1.0000
   
   -0.0475   -0.0103   -0.0001
   -0.0765   -0.8271   -0.0043
   55.3943  311.1104    1.0000
    """
    M['M1_7'] = np.array([[-0.0475, -0.0097, -0.0001], [-0.0765, -0.7754, -0.0043], [55.3943, 291.6660, 1.0000]])
    keys = list(M.keys())
    for k in keys:
        if not k.startswith('M'):
            M.pop(k, None)
    # match videos and M matrices
    videos = sorted([(i, x) for i, x in enumerate(sorted(os.listdir(video_dir))) if x.endswith('.mp4')])
    video_indices = dict()
    for i, v in videos:
        video_indices[v] = i
    print(video_indices)
    Ms = dict()
    for k in M.keys():
        m = M[k]
        t = k.replace('M', '').split('_')
        if len(t) == 1:
            s, e = t[0], t[0]
        else:
            s, e = t
        s, e = int(s), int(e)
        for i in range(s, e+1):
            Ms[videos[i][1]] = m
            
    # generate tracklet
    videonames = [x for x in os.listdir(video_dir) if x.startswith("Loc2")]
    for videoname in videonames:
        print(videoname)
        print("Processing video {}...".format(videoname))
        # detect_files
        pkl_files = sorted(os.listdir(os.path.join(detect_dir, videoname)))
        # tracking file
        tracklet = dict()
        tracklet_notrim = dict()
        track_file = videoname.replace('.mp4', '.txt')
        with open(os.path.join(track_dir, track_file)) as f:
            fnum = 0
            #pts = []
            track_ids = []
            for line in tqdm(f):
                data = line.split(',')
                fid, tid = int(data[0]), int(data[1])
                tx1, ty1, tw, th = [float(x) for x in data[2:6]]
                bx1, by1, bw, bh = [float(x) for x in data[6:10]]

                x1, y1 = int(bx1), int(by1) # xmin, ymin
                x2, y2 = int(x1 + bw), int(y1 + bh) # xmax, ymax
                margin = 50
                border_mark = False
                if not (x1 > margin and x2 < frame_w-margin and y1 > margin and y2 < frame_h-margin):
                    border_mark = True

                if fid > fnum:
                    # read new frame pickle
                    r = pickle.load(open(os.path.join(detect_dir, videoname, pkl_files[fid]), "rb"))
                    if videoname.startswith("Loc1_1"): # a tuple of (fnum, data)
                        r = r[1]
                    fnum = fid
                    #pts = []
                    track_ids = []

                # read mask roi
                boxes, masks = r['rois'], r['contours']
                index = np.where((r['rois'] == np.array([y1, x1, y2, x2])).all(axis=1))[0]
                if index.shape[0] == 1:
                    index = index[0]
                    c = r['contours'][index]
                    score = r['scores'][index]
                    bc = extract_bottom(c)
                else:
                    c = None
                    bc = None
                    continue

                # 2d to 3d
                m = Ms[videoname]
                bc = np.concatenate(bc, axis=0).reshape([-1, 1, 2]).squeeze() # N, 2
                bc3d = coord2dto3d(m, bc)
                #pts.append(bc3d)
                track_ids.append(np.array([tid]*bc3d.shape[0]))

                if not border_mark:
                    if tid in tracklet:
                        tracklet[tid].append([fid, y1, x1, y2, x2, score, bc, bc3d])
                    else:
                        tracklet[tid] = [[fid, y1, x1, y2, x2, score, bc, bc3d]]
                        
                if tid in tracklet_notrim:
                    tracklet_notrim[tid].append([fid, y1, x1, y2, x2, score, border_mark, bc, bc3d])
                else:
                    tracklet_notrim[tid] = [[fid, y1, x1, y2, x2, score, border_mark, bc, bc3d]]
        print('Finished Tracklet')
        # Speed estimation
        speed = dict()
        for track_id in tqdm(tracklet):
            fids = [pt3d[0] for pt3d in tracklet[track_id]]
            scores = np.expand_dims([pt3d[5] for pt3d in tracklet[track_id]], axis=1)
            bboxs = np.concatenate([np.array(pt3d[1:5]).reshape(1,4) for pt3d in tracklet[track_id]], axis=0)
            xy2d = np.concatenate([np.mean(pt3d[-2], axis=0, keepdims=True) for pt3d in tracklet[track_id]], axis=0)
            # interpolation
            frange = np.arange(min(fids), max(fids)+1)
            x2d = np.interp(frange, fids, xy2d[:, 0])
            y2d = np.interp(frange, fids, xy2d[:, 1])
            if frange.shape[0] < 32+11:
                continue
            
            x2d, y2d = smooth(x2d, window_len=11), smooth(y2d, window_len=11)
            xy2d_smooth = np.concatenate([np.expand_dims(x2d, axis=1), np.expand_dims(y2d, axis=1)], axis=1)
            m = Ms[videoname]
            xy3d_smooth = coord2dto3d(m, xy2d_smooth)
            x, y = xy3d_smooth[:, 0], xy3d_smooth[:, 1]
            # miles per hour
            v = (30*3600/1000.)*(np.diff(y)**2 + np.diff(x)**2)**0.5
            
            hour_smoothv = smoothv(v, window_len=31)[15:]
            mile_smoothv = 0.621371*hour_smoothv
            hvout = list(hour_smoothv[np.array(fids)-min(fids)])
            mvout = list(mile_smoothv[np.array(fids)-min(fids)])
            
            # append speed
            
            border_mark = [pt3d[6] for pt3d in tracklet_notrim[track_id]]
            if len(border_mark) > len(hvout):
                if border_mark[0]:
                    hvout = [hvout[0]]*(len(border_mark)-len(hvout)) + hvout
                    mvout = [mvout[0]]*(len(border_mark)-len(mvout)) + mvout
                elif border_mark[-1]:
                    hvout = hvout + [hvout[-1]]*(len(border_mark)-len(hvout))
                    mvout = mvout + [mvout[-1]]*(len(border_mark)-len(mvout))
                else:
                    print(videoname, track_id)
                    hvnew, mvnew = list(), list()
                    pt = 0
                    for b in border_mark:
                        if not b:
                            hvnew.append(hvout[pt])
                            mvnew.append(mvout[pt])
                            pt += 1
                        else:
                            hvnew.append(0.5*(hvout[pt]+hvout[pt-1]))
                            mvnew.append(0.5*(mvout[pt]+mvout[pt-1]))
                    hvout, mvout= hvnew, mvnew
            
            fids_notrim = np.expand_dims(np.array([pt3d[0] for pt3d in tracklet_notrim[track_id]]), axis=1)
            scores_notrim = np.expand_dims([pt3d[5] for pt3d in tracklet_notrim[track_id]], axis=1)
            bboxs_notrim = np.concatenate([np.array(pt3d[1:5]).reshape(1,4) for pt3d in tracklet_notrim[track_id]], axis=0)
            hvout_notrim = np.expand_dims(hvout, axis=1)
            mvout_notrim = np.expand_dims(mvout, axis=1)
            
            assert fids_notrim.shape[0] == scores_notrim.shape[0] == bboxs_notrim.shape[0] == hvout_notrim.shape[0] == mvout_notrim.shape[0], (fids_notrim.shape, scores_notrim.shape, bboxs_notrim.shape, hvout_notrim.shape, mvout_notrim.shape, len(border_mark), len(hvout), border_mark)
            speed[track_id] = np.concatenate([fids_notrim, scores_notrim, bboxs_notrim, hvout_notrim, mvout_notrim], axis=1)

        speed_list = list()
        for track_id in speed:
            for bbox in speed[track_id]:
                fid, score, y1, x1, y2, x2, hv, mv = bbox
                speed_list.append([int(fid), track_id, y1, x1, y2, x2, hv, mv, score])
        speed_list = sorted(speed_list, key=lambda x: x[0])
        print('Finished Speed')
        # save speed list
        with open(os.path.join(savetxt_dir, videoname.replace('.mp4','.txt')), 'w+') as f:
            for line in speed_list:
                fid, tid, y1, x1, y2, x2, hv, mv, score = line
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)
                video_id = video_indices[videoname]
                out = [video_id, fid+1, tid, xmin, ymin, xmax, ymax, mv, score]
                string = " ".join([str(round(x, 4)) for x in out])+"\n"
                f.write(string)

    # generate the final output file
    # savetxt_dir = "../../aic2018/track1/speed_txt"
    speed_txtfiles = sorted([x for x in os.listdir(savetxt_dir) if (x.startswith('Loc'))])
    cnt_dict = dict()
    outlier = []
    factor = 1.0
    with open(os.path.join(savetxt_dir, 'track1.txt'), 'w+') as f:
        for txt in speed_txtfiles:
            cnt = 0
            with open(os.path.join(savetxt_dir, txt), 'r') as ftxt:
                for line in ftxt:
                    vid, fid, tid, xmin, ymin, xmax, ymax, mv, score = line.split(' ')
                    tid = str(-1)
                    mv = str(float(mv) * factor)
                    out = [vid, fid, tid, xmin, ymin, xmax, ymax, mv, score]
                    string = ' '.join(out)
                    f.write(string)
                    cnt += 1
                print(txt, cnt)
                cnt_dict[txt] = cnt

        
if __name__ == '__main__':
    main()

    