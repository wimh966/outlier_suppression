# Outlier Suppression framework

PyTorch implementation of Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models

## Overview
The outlier suppression framework effectively suppresses the outliers for transformer language models to pursue superiority in quantization. It contains two components: Gamma Migration and Token-Wise Clipping. 

The framework can be adopted in both PTQ and QAT pipelines and can be applied to different models and tasks. We are giving examples here to help run our framework on BERT, RoBERTa, and BART models across text classification, question answering, and summarization tasks. 

## Introduction of config

Firstly, we illustrate some brief descriptions about our config for better understanding.

config.yaml

```
quant: 
    is_remove_padding: True
    calibrate: 256
    ln: 
        delay: False
    a_qconfig:
        quantizer: FixedFakeQuantize 
        observer: AvgMinMaxObserver
        bit: 6
        symmetric: False
        ch_axis: -1
    w_qconfig:
        quantizer: FixedFakeQuantize
        observer: MinMaxObserver
        bit: 6
        symmetric: True
        ch_axis: 0
```
This is an example of MinMax Quantization in our config. We put the explanation about each item below, and also clarify other quantization settings.

* is_remove_padding (bool) - whether remove pad token during calibration: False | True.  Because pad token will not influence the FP accuracy, it naturally needs to be removed during calibration to exert no influence on the quantization parameters.  Default: True
* calibrate (int) - the number of calibration examples. Default: 256
* ln: config about Gamma Migration
  * delay (bool) - whether activate Gamma Migration component: False | True. Default: False
* a_qconfig/ w_qconfig - config about quantization scheme on activation or weight
  * bit (int) - quantization bit
  * symmetric (bool) - whether use symmetric quantization: False | True. Default: for weight: False, for activation: True
  * ch_axis (int) - per-tensor or per-channel quantization: -1 | 0. -1: per-tensor quantization, 0: per-channel quantization at dim 0. Default: -1/0
  * quantizer (string, optional) - quantizer type: FixedFakeQuantize | LSQFakeQuantize | LSQPlusFakeQuantize. FixedFakeQuantize: normal quantizer, LSQFakeQuantize: mark scale as the parameter, LSQPlusFakeQuantize: mark scale and zero-point as parameters. Default: FixedFakeQuantize
  * observer (string, optional) - collect activation/ weight statistics to identify the initiliazed scale and zero-point. For activation: AvgMinMaxObserver | AvgMSEFastObserver | AvgQuantileObserver | AvgPruneMinMaxObserver. AvgPruneMinMaxObserver: activate the coarse-grained phase of Token-Wise Clipping. Default: AvgMinMaxObserver. For weight: MinMaxObserver | MSEFastObserver | LSQPlusObserver. Default: MinMaxObserver. Usually, for 6/8-bit weight, MinMaxObserver is enough, for lower bit (4-bit), suggest MSEFastObserver or LSQPlusObserver (only suitable in QAT).


Therefore, we can get the config of our outlier suppression framework below. Set delay to True to enable the Gamma Migration. Set observer in activation as AvgPruneMinMaxObserver to enable the coarse-grained phase of Token-Wise Clipping. Set quantizer in activation as LSQPlusFakeQuantize to additionally do the fine-grained stage of Token-Wise Clipping.
```
quant:
    is_remove_padding: True
    calibrate: 256
    ln: 
        delay: True
    a_qconfig:
        quantizer: LSQPlusFakeQuantize 
        observer: AvgPruneMinMaxObserver
        bit: 6
        symmetric: False
        ch_axis: -1
    w_qconfig:
        quantizer: FixedFakeQuantize
        observer: MinMaxObserver
        bit: 6
        symmetric: True
        ch_axis: 0
```
## Usage
### Env
Install huggginface (version 4.18.0) and datasets (version 1.17.0) in advance.
Use Tesla V100 to run the code.
### Data & Model
Download the data (metric) in advance, and put them in your ~/.cache/huggingface directory. Details can be found in doc of huggingface. Or you can run the code directly. It will download the data automatically.

We provide the fine-tuned FP32 models in https://huggingface.co/ModelTC. You can download and put them in your own directory.
### PTQ (post-training quantization)
We give the config for each task in exp directory.

#### GLUE
Go into the exp/bert_ptq directory. It contains different directories indicating different calibration methods.

Now, let's look at cola task in the twc_fine_gamma (Ours) directory. You need to put your model path in 'model_name_or_path' in config.yaml. Then you can run the run.sh and get the result.

Moreover, for RoBERTa and BART models, replace the model path in 'model_name_or_path' in config.yaml.

It will run ptq_glue_quant.py

#### SQuAD
Go into the exp/bert_ptq directory. Other operations are the same as GLUE.

It will run ptq_qa_quant.py

#### CNN/DailyMail & XSum
Go into the exp/cnn_dailymail (exp/xsum) directory. Replace the model path to yours in item model_name_or_path in the config. Then you can run the run.sh and get the result.

It will run ptq_summ_quant.py.

### QAT (quantization-aware training)
We are still sorting out the code of the part. Looking forward to it!
