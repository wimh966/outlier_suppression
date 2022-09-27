#!/usr/bin/env bash
PYTHONPATH=$PYTHONPATH:../../../../ GLOG_vmodule=MemcachedClient=-1 \
python ../../../../quant_transformer/solver/ptq_qa_quant.py --config config.yaml
