import yaml
import os
from easydict import EasyDict
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


def load_dataset_labels(config_data):
    # datasets
    raw_datasets = load_dataset("glue", config_data.task_name)
    # num_labels
    if config_data.is_regression:
        num_labels = 1
        label_list = None
    else:
        label_list = raw_datasets['train'].features['label'].names
        num_labels = len(label_list)
    return raw_datasets, num_labels, label_list


def load_model(config_model, config_data, num_labels):
    # num_labels first to indentity the classification heads
    tokenizer = AutoTokenizer.from_pretrained(
        config_model.tokenizer_name if config_model.tokenizer_name else config_model.model_name_or_path,
        cache_dir=config_model.cache_dir,
        use_fast=config_model.use_fast_tokenizer,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    config_tmp = AutoConfig.from_pretrained(
        config_model.config_name if config_model.config_name else config_model.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=config_data.task_name,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    if hasattr(config_model, 'attn_dropout'):
        config_tmp.attention_probs_dropout_prob = config_model.attn_dropout
    if hasattr(config_model, 'hidden_dropout'):
        config_tmp.hidden_dropout_prob = config_model.hidden_dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        config_model.model_name_or_path,
        from_tf=bool(".ckpt" in config_model.model_name_or_path),
        config=config_tmp,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    return tokenizer, model


def preprocess_dataset(config_data, training_args, raw_datasets, label_to_id, tokenizer):
    # tokenize the data
    sentence1_key, sentence2_key = task_to_keys[config_data.task_name]
    if config_data.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False
    max_seq_length = config_data.max_seq_length

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not config_data.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return raw_datasets


def check_return_data(raw_datasets, data_type, do_type, max_samples):
    if not do_type:
        return None
    if do_type and data_type not in raw_datasets:
        raise ValueError(f'do- {data_type} requires a {data_type} dataset')
    type_dataset = raw_datasets[data_type]
    if max_samples is not None:
        type_dataset = type_dataset.shuffle().select(range(max_samples))
    return type_dataset
