import skvideo.io
import os
import argparse
import time
import cv2
import math
import numpy as np

def video2frame(video_name, save_path):
    
    print("Reading {} and saving at {}".format(video_name, save_path))
    cap = skvideo.io.vreader(video_name)
    #cap = cv2.VideoCapture(video_name)
    #cap.set(cv2.CAP_PROP_FPS, 20)

    # Check if camera opened successfully
    #if (cap.isOpened()== False): 
    #    print("Error opening video stream or file")

    # retrieve video info
    #fps = cap.get(cv2.CAP_PROP_FPS)
    vlen = int(skvideo.io.FFmpegReader(video_name).getShape()[0])
    #print("Video: {}, Video length: {}, FPS: {}".format(video_name.split("/")[-1], vlen, fps))
    vn = video_name.split('\\')[1].split('.')[0]
    cnt = 0
    start_time = time.time()
    """
    ret = True
    while(ret):
        ret, frame = cap.read()
        if ret:
            image_name = str(video_files.index(video_name)) + "_" + str(cnt).zfill(math.ceil(np.log10(vlen))) + ".jpg"
            cv2.imwrite(os.path.join(save_path, image_name), frame)
            cnt += 1
        elif cnt < vlen:
            raise ValueError("Video {} abort because some frames are missing".format(video_name.split("/")[-1]))
        else:
            break  
    """
    for frame in cap:
        image_name = vn + "_" + str(cnt).zfill(math.ceil(np.log10(vlen))) + ".jpg"
        cv2.imwrite(os.path.join(save_path, image_name), frame[:,:,::-1])
        cnt += 1
    
    print("Finsh video {} in {:.4f} seconds.".format(video_name.split("/")[-1], time.time()-start_time)) 
   
    return


def main(vedio_dir, save_path):
    for v in os.listdir(vedio_dir):
        file_save_path = os.path.join(save_path,v.split('.')[0],'img1')
        if os.path.exists(file_save_path):
            print('Video',v,'already changed to frames.')
            continue
        else:
            os.makedirs(file_save_path)
        video2frame(os.path.join(vedio_dir,v),file_save_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Video2Frame")
    parser.add_argument(
        "--vedio_dir", help="Path the videos",
        default='../data/track1_videos')
    parser.add_argument(
        "--save_path", help="Path to save the images.", 
        default='../data/Nvidia')
    args = parser.parse_args()
    main(args.vedio_dir,args.save_path)
    
    
    