# Track 1

## Dependency

To install all dependency packages, execute `$ pip install -r requirements.txt `

- tensorfow
- numpy
- opencv
- imageio
- h5py
- scikit-learn
- scipy
- matplotlib

## Models & Data

- Download pretrained models from [here](). Upzip "track1_model.zip" and put it under "./model/".
- Download detection output data from [here](). It contains all detection output files provided by Mask-RCNN. Upzip "track1_data.zip" and put it under "./data/".

## Steps

### Vehicle detection

1. If want to re-generate all detection outputs from , go to "./detect/" and run "detect_pkl.py" which will save all detection results of each video frame as a pickle file. The default saving directory is "../data/detect_output_pkl/".

   `python aic_video_detect.py`

2. Prepare 

### Vehicle tracking

1. Move to "./track/" directory.

2. Generate the feature vectors of each detection bounding box by a pretrained appearance model.

   `python tools/generate_detections.py --model=../model/cosine/detrac.pb --mot_dir=../data/detection_output_txt/Nvidia --output_dir=../data/track_features/Nvidia`

3. Generate tracking sequences by running "deep_sort_app.py". It will save result as "../output/track/".

   `python deep_sort_app.py --sequence_dir=./Nvidia --detection_file=./resources/detections/Nvidia --output_dir=../output/track --min_confidence=0.3 --nn_budget=100 --display=0`

### Speed

1. Go to "./speed" directory.
2. Run "speed.py" to implement speed estimation from tracking trajectories. The final output "track1.txt" will be saved in "../output/track1.txt".