# Track 1

## Dependency

To install all dependency packages, execute `$ pip install -r requirements.txt `

- tensorfow
- numpy
- opencv
- imageio
- h5py
- scikit-learn
- Scikit-video
- scipy
- matplotlib

## Models & Data

- Download pretrained models and from detection output data [here](https://drive.google.com/open?id=1RH3nObVsXTynVU2Gd3erD_cd9TJ1qYWs). It contains all detection output files provided by Mask-RCNN.
- Upzip "model.zip" and "data.zip".

## Steps

### Vehicle detection

1. If want to re-generate all detection outputs from , go to "./detect/" and run "detect_pkl.py" which will save all detection results of each video frame as a pickle file. The default saving directory is "../data/detect_output_pkl/".

   `python aic_video_detect.py --regenerate True `

2. Prepare txt files for tracking.

    `python pkl_to_det.py --video_dir ../data/track1_videos/ --detection_file ../data/detect_output_pkl/ --output_dir ../data/detect_output_txt` 

### Vehicle tracking

1. Move to "./track/" directory.

2. Convert videos to images. `python tools/Video2Frames.py`

3. If you are not using tensorflow1.4.0, then ``. This command could help to convert model into 

4. Generate the feature vectors of each detection bounding box by a pretrained appearance model.

   `python tools/generate_detections.py --model=../model/cosine/detrac.pb --mot_dir=../data/detection_output_txt/Nvidia --output_dir=../data/track_features/Nvidia`

5. Generate tracking sequences by running "deep_sort_app.py". It will save result as "../output/track/".

   `python deep_sort_app.py --sequence_dir=./Nvidia --detection_file=./resources/detections/Nvidia --output_dir=../output/track --min_confidence=0.3 --nn_budget=100 --display=0`

### Speed

1. Go to "./speed" directory.
2. Run "speed.py" to implement speed estimation from tracking trajectories. The final output "track1.txt" will be saved in "../output/speed_txt/track1.txt".

## Reference

1. He, Kaiming, et al. "Mask r-cnn." *Computer Vision (ICCV), 2017 IEEE International Conference on*. IEEE, 2017.
2. Wojke, Nicolai, Alex Bewley, and Dietrich Paulus. "Simple online and realtime tracking with a deep association metric." *arXiv preprint arXiv:1703.07402* (2017).
3. https://github.com/matterport/Mask_RCNN
4. https://github.com/nwojke/deep_sort