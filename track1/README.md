# Track 1

## Dependency

* tensorfow
* numpy
* opencv
* imageio
* pickle
* h5py
* ​

To install all dependency packages, execute `$ pip install -r requirements.txt `



## Data

Download detection output data from [here]() and upzip "data.zip". It contains all detection output files provided by Mask-RCNN.

## Vehicle detection

If want to re-generate all detection outputs from , go to "./detect/" and run "detect_pkl.py" which will save all detection results of each video frame as a pickle file. The default saving directory is.

## Vehicle tracking

1. Move to "./track/" directory.
2. Generate the feature vectors of each detection bounding box by a pretrained appearance model. To implement this step, run "xxx.py".
3. Generate tracking sequences by running "track.py". It will save result in "xxx".

## Speed

1. Go to "./speed" directory.
2. Run "speed.py" to implement speed estimation from tracking trajectories. The final output "track1.txt" will be saved in "../output/track1.txt".