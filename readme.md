# Outlier Suppression framework

Official PyTorch implementation of  <a href="https://arxiv.org/abs/2209.13325" >Outlier Suppression: Pushing the Limit of Low-bit Transformer</a>, NeurIPS 2022.


## Overview
The outlier suppression framework effectively suppresses the outliers for transformer language models to pursue superiority in quantization. It contains two components: Gamma Migration and Token-Wise Clipping. 

The framework can be adopted in both post-training quantization (PTQ) and quantization-aware training (QAT) pipelines and can be applied to different models and tasks. We are giving examples here to help run our framework on BERT, RoBERTa, and BART models across text classification, question answering, and summarization tasks. 


## Usage
### Env
Install huggginface (version 4.18.0) and datasets (version 1.17.0) in advance.
Use Tesla V100 to run the code.

### Data & Model
Download the data (metric) in advance, and put them in your ~/.cache/huggingface directory. Details can be found in doc of huggingface. Or you can run the code directly. It will download the data automatically.

We provide the fine-tuned FP32 models in https://huggingface.co/ModelTC. You can download and put them in your own directory.

### Experiments on PTQ
We give the config for each task in exp directory.
```
exp/
├── bert_ptq
├── xsum
└── cnn_dailymail
```
### GLUE (Text classification tasks)
We take bert models as an example. For RoBERTa and BART, you only need to change the model_path ('model_name_or_path') in config.yaml.

Configuration of BERT models are:
```
exp/bert_ptq/
├── twc_fine_gamma
│   └── cola
│   │   └── config.yaml
│   │   └── run.sh
│   └── mnli
│   └── ...
│   └── stsb
├── minmax
├── quantile
└── mse
```

Take an example of running CoLA task using our methods, others are the same. Run the following and you can get the results.
```
cd exp/bert_ptq/twc_fine_gamma/cola
bash run.sh
```

### SQuAD (Question answering tasks)
We take bert models as an example. For RoBERTa and BART, you only need to change the model_path ('model_name_or_path') in config.yaml.

Configuration of BERT models are:
```
exp/bert_ptq/twc_fine_gamma
├── squad_v1
│   └── config.yaml
│   └── run.sh
└── squad_v2
```
Run the following and you can get the result.

```
cd exp/bert_ptq/twc_fine_gamma/squad_v1
bash run.sh
```

### CNN/DailyMail & XSum (Summarization tasks)
```
exp/xsum/twc_fine_gamma
├── config.yaml
└── run.sh
```
Run the following and you can get the result.
```
cd exp/xsum/twc_fine_gamma
bash run.sh
```
### Experiments on QAT
We are still sorting out the code of the part. Looking forward to it!


## Introduction of config

We put some brief descriptions about the config for better understanding.

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
This is an example of MinMax Quantization in our config. Below is the explanation about each item.

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


Based on these, below is the config of our outlier suppression framework. Set delay to True to enable the Gamma Migration. Set observer in activation as AvgPruneMinMaxObserver to enable the coarse-grained phase of Token-Wise Clipping. Set quantizer in activation as LSQPlusFakeQuantize to additionally do the fine-grained stage of Token-Wise Clipping.
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

## Reference

If you find this repo useful for your research, please consider citing the paper:
```
    @article{wei2022outlier,
    title={Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models},
    author={Wei, Xiuying and Zhang, Yunchen and Zhang, Xiangguo and Gong, Ruihao and Zhang, Shanghang and Zhang, Qi and Yu, Fengwei and Liu, Xianglong},
    journal={arXiv preprint arXiv:2209.13325},
    year={2022}
    }
```