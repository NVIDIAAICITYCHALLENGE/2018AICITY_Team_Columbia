import skvideo.io
import os
import shutil
import argparse
import time
import cv2
import math
import numpy as np
import multiprocessing


def video2frame(video_name, save_path):
    
    print("Reading {} and saving at {}".format(video_name, save_path))
    cap = skvideo.io.vreader(video_name)
    
    vlen = int(skvideo.io.FFmpegReader(video_name).getShape()[0])
    vn = os.path.basename(video_name)
    assert vn.endswith('.mp4'), "{} is not a video".format(vn)
    vn = vn.replace('.mp4', '')
    cnt = 0
    start_time = time.time()
    
    for frame in cap:
        image_name = vn + "_" + str(cnt).zfill(math.ceil(np.log10(vlen))) + ".jpg"
        cv2.imwrite(os.path.join(save_path, image_name), frame[:,:,::-1])
        cnt += 1
    
    print("Finsh video {} in {:.4f} seconds.".format(video_name.split("/")[-1], time.time()-start_time)) 
   
    return


def main(video_dir, save_path):
    jobs = []
    video_names = [x for x in os.listdir(video_dir) if x.startswith('Loc')]
    for v in video_names:
        file_save_path = os.path.join(save_path,v.split('.')[0],'img1')

        # if os.path.exists(file_save_path):
        #     shutil.rmtree(file_save_path)
        #     print('Remove {}'.format(file_save_path))

        if os.path.exists(file_save_path):
            if not os.listdir(file_save_path) == list():
                print(file_save_path)
                print(len(os.listdir(file_save_path)))
                print('Video',v,'already changed to frames.')
                continue
        else:
            os.makedirs(file_save_path)
        jobs.append(multiprocessing.Process(target=video2frame,
                                            args =(os.path.join(video_dir,v),file_save_path)))
    
    # start jobs
    for j in jobs:
        j.start()
 
    # make sure all jobs have finished
    for j in jobs:
        j.join()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Video2Frame")
    parser.add_argument(
        "--video_dir", help="Path the videos",
        default='../data/track1_videos')
    parser.add_argument(
        "--save_path", help="Path to save the images.", 
        default='../data/Nvidia')
    args = parser.parse_args()
    print(os.getcwd())
    main(os.path.join(os.getcwd(), args.video_dir), os.path.join(os.getcwd(), args.save_path))
    
    
    
