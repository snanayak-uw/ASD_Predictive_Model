# ASD_Predictive_Model
===========================================================================

Description

This folder contains the files necessary necessary to build a predictive 
model for ASD based on connectivity in the brain. Preprocessed 
resting-state fMRI data is retreived from ABIDE I dataset and connectivity 
is calculated using the nilearn package in Python. Connectivity matrices 
and phenotypic data is exported to MATLAB for further analysis. A 
predictive model is built using L1-SCCA in feature selection sparse 
logistic regression as described by Yahata et al. (2016). 

===========================================================================


To run the code that builds the predictive model, see: 
PredictiveModel.m
Note: Run MATLAB_prep.ipynb before running model code for the first time

To understand the contents of this folder, see: 
File_Descriptions.docx

To use the determined coefficients to predict diagnosis on new data, see:
FinalCoefficients.mat

To fetch the original data from the ABIDE dataset, see:
MATLAB_prep.ipynb

===========================================================================

Additional resources:

Python package for fMRI data analysis:
https://nilearn.github.io/index.html

MATLAB Toolbox for sparse logistic regression:
https://bicr.atr.jp/~oyamashi/SLR_WEB.html

Research paper explaining results and methodology:
Noriaki Yahata, Jun Morimoto, Ryuichiro Hashimoto, Giuseppe Lisi, Kazuhisa 
Shibata, Yuki Kawakubo, Hitoshi Kuwabara,Miho Kuroda, Takashi Yamada, 
Fukuda Megumi, Hiroshi Imamizu, Jose E. Nanez, Hidehiko Takahashi, Yasumasa
Okamoto, Kiyoto Kasai, Nobumasa Kato, Yuka Sasaki, Takeo Watanabe, and 
Mitsuo Kawato (2016). A Small Number of Abnormal Brain Connections Predicts
Adult Autism Spectrum Disorder, NatComms
