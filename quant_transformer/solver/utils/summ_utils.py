import yaml
import os
from easydict import EasyDict
import logging
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset
summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}
logger = logging.getLogger("transformer")


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


def load_model(config_model, config_data):
    config = AutoConfig.from_pretrained(
        config_model.config_name if config_model.config_name else config_model.model_name_or_path,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config_model.tokenizer_name if config_model.tokenizer_name else config_model.model_name_or_path,
        cache_dir=config_model.cache_dir,
        use_fast=config_model.use_fast_tokenizer,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config_model.model_name_or_path,
        from_tf=bool(".ckpt" in config_model.model_name_or_path),
        config=config,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < config_data.max_source_length
    ):
        if config_model.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {config_data.max_source_length}."
            )
            model.resize_position_embeddings(config_data.max_source_length)
        elif config_model.resize_position_embeddings:
            model.resize_position_embeddings(config_data.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {config_data.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    def use_task_specific_params(model, task):
        """Update config with summarization specific params."""
        task_specific_params = model.config.task_specific_params

        if task_specific_params is not None:
            pars = task_specific_params.get(task, {})
            logger.info(f"setting model.config to task specific params for {task}:\n {pars}")
            logger.info("note: command line args may override some of these")
            model.config.update(pars)
    use_task_specific_params(model, config_data.task)
    logger.info(f"New config {model.config}")
    return tokenizer, model


def load_summ_dataset(config_model, config_data):
    if config_data.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            config_data.dataset_name,
            config_data.dataset_config_name,
            cache_dir=config_data.cache_dir,
            use_auth_token=True if config_model.use_auth_token else None,
        )
    else:
        data_files = {}
        if config_data.train_file is not None:
            data_files["train"] = config_data.train_file
            extension = config_data.train_file.split(".")[-1]
        if config_data.validation_file is not None:
            data_files["validation"] = config_data.validation_file
            extension = config_data.validation_file.split(".")[-1]
        if config_data.test_file is not None:
            data_files["test"] = config_data.test_file
            extension = config_data.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=config_data.cache_dir,
            use_auth_token=True if config_data.use_auth_token else None,
        )
    'update data config, assert dataset name is in summarization_name_mapping'
    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(config_data.dataset_name, None)
    config_data.text_column = dataset_columns[0]
    config_data.summary_column = dataset_columns[1]
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


def preprocess_function(examples, tokenizer, config_data, max_target_length):
    # remove pairs where at least one record is None
    prefix = config_data.source_prefix if config_data.source_prefix is not None else ""
    padding = "max_length" if config_data.pad_to_max_length else False
    inputs, targets = [], []
    for i in range(len(examples[config_data.text_column])):
        if examples[config_data.text_column][i] is not None and examples[config_data.summary_column][i] is not None:
            inputs.append(examples[config_data.text_column][i])
            targets.append(examples[config_data.summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=config_data.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and config_data.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
