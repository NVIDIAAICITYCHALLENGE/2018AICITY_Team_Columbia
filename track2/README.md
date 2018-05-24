There are several steps to implement anomaly detection algorithms. Some of them require external github repo. 

1. The first step is to extract VGG features for eachm frame in a video sequence. I follow this [repo](https://github.com/ry/tensorflow-vgg16) of VGG 16 with pretrained weights to act as feature extractor at frame level. The length of extacted features is 4096, which is the second last layer of VGG 16.

2. Then, in the second step, we train a PCA reductor to reduce the dimension. The code can be found in `train_pca.py`. The reduced dimensionality of features is 256. If you do not have data to train such a PCA, you can directly use the trained PCA pickle file which is `pca.pkl`. The corresponding code for dimensionality reduction is in `pca_data_save.py`.

3. To encode the features, we use `vlfeat` library, which is in `vlfeat` folder. The source can be found [here](http://www.vlfeat.org/). To use VLAD encoding, we first use `kmeans.m` to cluster all the fatures. Then, we are ready to use `vlad_enc.m` to encode all the fleatures using VLAD encoding. We save the results into csv file.

4. To include temporal relationship, we extract motion features using iDT extractor as described [here](http://lear.inrialpes.fr/~wang/improved_trajectories).The prerequisite library is [`ffmpeg-0.11.1`](https://lear.inrialpes.fr/people/wang/download/ffmpeg-0.11.1.tar.bz2) and [`OpenCV-2.4.2`](https://lear.inrialpes.fr/people/wang/download/OpenCV-2.4.2.tar.bz2). After successfully installing both libraries, you can compile the code in `improved_trajectory_release` folder to generate excutable file. Then, you can save the results in csv file as well. Simlarily, we use VLAD encoding to encode this as well. 

5. After step 3 and 4, we have both features from frame level and temporal level. Then we can insert features into the trained pickle files to classify. The trained classifier can be found in the `Model` folder. 