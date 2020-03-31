# HECS

## Requirements:

* Tensorflow-gpu == 1.0.0

* Python 3.6

* CUDA 8.0 (For GPU)

## Introduction

This is the implementation of the paper "Hierarchical Embedding for Code Search in Software Q&A Sites".

## File
 
* CodeATTENTION.py  : The implementation of CodeATTENTION.

* CodeONLSTM.py : The implementation of CodeONLSTM.

* HECS.py : The implementation of HECS.

* attention_visualization.py : The implementation of drawing heatmap of attention.

* data_helper.py : Data helpy function for this experiments.

* main.py ：The Main function of this experiments.

* model01_base_LSTM_CNN.py ：The implementation of CodeRCNN

* model02_base_CNN.py ：The implementation of CodeCNN
 
* model03_base_LSTM.py ：The implementation of CodeLSTM
 
* model04_base_CODEnn.py ：The implementation of DeepCS

* model_utils.py  The metrics of experiments

## Details
We provide example codes to repeat the experiments.

```
$ python main.py --train=True
```
Selecting this parameter means training the model.The results (Recall@1 and Mean Reciprocal Rank (MRR)) of each training epoch are given during the training. If you want to get other results (FRank, Precision@k) you need to modify the evaluation function (eval_metric_rec1) of the main.py to "eval_metric".

```
$ python main.py --pred=True
```
Selecting this parameter means testing the existing model.The results (Recall@1 and Mean Reciprocal Rank (MRR)) of each training epoch are given during the training. If you want to get other results (FRank, Precision@k) you need to modify the evaluation function (eval_metric_rec1) of the main.py to "eval_metric".

```
$ python main.py --pmap=True
```
Selecting this parameter means drawing heatmap of attention based on existing models.

