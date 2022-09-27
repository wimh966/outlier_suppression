import os
import numpy as np
import logging
import sys
import argparse
import transformers
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
)
import datasets
import random
from datasets import load_metric
import torch  # noqa E401
import torch.fx
import quant_transformer.solver.utils.glue_utils as glue_utils
from quant_transformer.quantization.state import enable_calibration_woquantization, enable_quantization,\
        disable_all, enable_calibration_quantization, set_observer_name  # noqa: F401
from quant_transformer.quantization.observer import ObserverBase  # noqa: F401
from quant_transformer.quantization.fake_quant import LSQPlusFakeQuantize, QuantizeBase  # noqa: F401
from quant_model import quantize_model
import token_wise_clipping
logger = logging.getLogger("transformer")


def set_logger(config_progress):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = config_progress.log_level
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def make_huggingface_training_args(config_train, config_progress):
    training_args = TrainingArguments(
        seed=config_train.seed,
        output_dir=config_train.output_dir,
        overwrite_output_dir=config_train.overwrite_output_dir,
        do_train=config_train.do_train,
        do_eval=config_train.do_eval,
        do_predict=config_train.do_predict,
        evaluation_strategy=config_train.evaluation_strategy,
        eval_steps=config_train.eval_steps,
        per_device_train_batch_size=config_train.per_device_train_batch_size,
        per_device_eval_batch_size=config_train.per_device_eval_batch_size,
        gradient_accumulation_steps=config_train.gradient_accumulation_steps,
        eval_accumulation_steps=config_train.eval_accumulation_steps,
        learning_rate=config_train.learning_rate,
        weight_decay=config_train.weight_decay,
        max_grad_norm=config_train.max_grad_norm,
        num_train_epochs=config_train.num_train_epochs,
        max_steps=config_train.max_steps,
        lr_scheduler_type=config_train.lr_scheduler_type,
        warmup_ratio=config_train.warmup_ratio,
        warmup_steps=config_train.warmup_steps,
        gradient_checkpointing=config_train.gradient_checkpointing,
        log_level=config_progress.log_level,
        log_level_replica=config_progress.log_level_replica,
        logging_dir=config_progress.logging_dir,
        logging_strategy=config_progress.logging_strategy,
        logging_steps=config_progress.logging_steps,
        save_strategy=config_progress.save_strategy,
        save_steps=config_progress.save_steps,
        save_total_limit=config_progress.save_total_limit,
        save_on_each_node=config_progress.save_on_each_node,
        no_cuda=config_progress.no_cuda,
        run_name=config_progress.run_name,
        disable_tqdm=config_progress.disable_tqdm,
        load_best_model_at_end=config_progress.load_best_model_at_end,
        metric_for_best_model=config_progress.metric_for_best_model,
        greater_is_better=config_progress.greater_is_better
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    config_progress.log_level = training_args.get_process_log_level()
    return training_args


def prepare_input_output(trainer, cali_data):
    logger.info('**prepare fp input and output**')
    data_loader = trainer.get_eval_dataloader(cali_data)
    fp_input, fp_output = [], []
    with torch.no_grad():
        for p in data_loader:
            tmp = {}
            for k, v in p.items():
                tmp[k] = v.cuda()
            del tmp['labels']
            output = trainer.model(**tmp)[0].detach()
            fp_input.append(tmp)
            fp_output.append(output)
    return fp_input, fp_output


def calibrate(trainer, fp_input):
    logger.info("*** Calibrate ***")
    with torch.no_grad():
        for batch in fp_input:
            trainer.model(**batch)


def evaluate(trainer, eval_datasets):
    logger.info("*** Evaluate ***")
    if not isinstance(eval_datasets, tuple):
        eval_datasets = [eval_datasets]
    metrics = []
    for i in range(len(eval_datasets)):
        metric = trainer.evaluate(eval_dataset=eval_datasets[i])
        metrics.append(metric)
    for i in range(len(metrics)):
        trainer.log_metrics("eval", metrics[i])
        trainer.save_metrics("eval", metrics[i])


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config_path):
    config = glue_utils.parse_config(config_path)
    set_seed(config.train.seed)
    if config.data.task_name == 'cola':
        config.progress.metric_for_best_model = 'matthews_correlation'
    elif config.data.task_name == 'stsb':
        config.progress.metric_for_best_model = 'pearson'
    else:
        config.progress.metric_for_best_model = 'accuracy'
    training_args = make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    raw_datasets, num_labels, label_list = glue_utils.load_dataset_labels(config.data)
    tokenizer, model = glue_utils.load_model(config.model, config.data, num_labels)

    # label2id & id2label
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and config.data.task_name is not None
        and config.data.task_name != 'stsb'
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    elif config.data.task_name is not None and config.data.task_name != 'stsb':
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    # max_seq_length
    config.data.max_seq_length = min(config.data.max_seq_length, tokenizer.model_max_length)

    # work with datasets, preprocess first then get train/val/test one
    raw_datasets = glue_utils.preprocess_dataset(config.data, training_args, raw_datasets, label_to_id, tokenizer)
    # train_dataset, val_dataset, predict_datasets
    train_datasets = glue_utils.check_return_data(raw_datasets, 'train', True, config.data.max_train_samples)

    if config.data.task_name == 'mnli':
        eval_datasets = (
            glue_utils.check_return_data(raw_datasets, 'validation_matched', True, config.data.max_eval_samples),
            glue_utils.check_return_data(raw_datasets, 'validation_mismatched', True, config.data.max_eval_samples),
        )
    else:
        eval_datasets = (glue_utils.check_return_data(raw_datasets, 'validation', True, config.data.max_eval_samples), )

    metric = load_metric("glue", config.data.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if config.data.is_regression else np.argmax(preds, axis=1)

        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if config.data.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    model.eval()
    model.cuda()
    if getattr(config, 'quant', None):
        fp_model = model
        fp_model.eval()
        model = quantize_model(model, config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets[0],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    if getattr(config, 'quant', None):
        cali_data = train_datasets.shuffle().select(range(config.quant.calibrate))
        fp_input, fp_output = prepare_input_output(trainer, cali_data)
        if config.quant.ln.delay:
            from gamma_migration import delay_ln
            trainer.model = delay_ln(trainer.model, config.quant, config.model)
        # calibrate the weight
        enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
        calibrate(trainer, [fp_input[0]])
        if 'PruneMinMaxObserver' in config.quant.a_qconfig.observer:
            disable_all(trainer.model)
            set_observer_name(trainer.model)
            token_wise_clipping.token_wise_clipping(trainer, fp_input, fp_output, config)
            if 'LSQ' in config.quant.a_qconfig.quantizer:
                token_wise_clipping.learn_scale(trainer, fp_input, fp_output,
                                                getattr(config.quant, 'learn', {'lr': 1e-5, 'epoch': 3}))
        else:
            # calibrate the activation
            enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
            calibrate(trainer, fp_input)
        torch.cuda.empty_cache()

    if training_args.do_eval:
        if getattr(config, 'quant', None):
            enable_quantization(trainer.model)
        torch.cuda.empty_cache()
        evaluate(trainer, eval_datasets)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
