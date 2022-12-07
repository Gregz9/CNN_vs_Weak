# CNN_vs_Weak

This repo contains code and report related to project 3 in subject FYS-STK3155/FYS_STK4155 - Applied Dataanalysis and Machine Learning at University of Oslo. The main focus of this research is to compare strong and weak classifiers ability to classify images of pneumonia.

Assess dimention reduction methods combined with weak and strong learners for prediction of pneumonia. We will investigate PCA and Convolution for image dimentionality reduction (and DCT JPEG dimentionality reduction if we have time), combined with neural networks and random forests (and XGBoost if we have time).

## Instructions on retrieving data

We use Chest X-Ray images of from [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2). To download these images, and create the necessary file structure, you can run the shell script ```downloadData.sh```, if you are on linux or OSX. If you are on windows, you can download the data [here](https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded). After this, create a directory called ```data```, place ```ChestXRay2017.zip``` there and unzip it.

## Ting som må gjøres:

### MNIST
- [x] CONV NN er bra
- [ ] Conv RF kan kanskje skrives mer på
- [x] PCA RF er decent
- [x] PCA NN er decent

### PNEU
- [x] CONV NN
- [ ] Conv RF er gjort men dårlig og ikke skrevet
- [ ] PCA RF er ikke gjort
- [ ] PCA NN er gjort men må skrives

### Rapport
- [ ] Abstract
- [x] Intro
- [x] Theory
- [x] Method
- [ ] Results
- [ ] Conclusion

### Andre ting
- [ ] Precision og recall
- [ ] Data augmentation
