# CNN_vs_Hybrid_Models

This repo contains code and report related to project 3 in subject FYS-STK3155/FYS_STK4155 - Applied Data Analysis and Machine Learning at University of Oslo. The main focus of this research is to compare CNNs to hybrid models, exploring whether simpler models can perform similarly to CNNs at a lower computational cost. The paper is located at ```doc/paper.pdf```, while the source code is located at ```src```.

## Instructions on retrieving data

We use Chest X-Ray images of from [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2). To download these images, and create the necessary file structure, you can run the shell script ```downloadData.sh```, if you are on linux or OSX. If you are on windows, you can download the data [here](https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded). After this, create a directory called ```data```, place ```ChestXRay2017.zip``` there and unzip it.

## Dependencies
```
tensorflow
tensorflow-decision-forests
keras-tuner
numpy
matplotlib
seaborn
scikit-learn
```
