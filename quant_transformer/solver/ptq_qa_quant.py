import numpy as np
import os
import sys
import torch
import random
import logging
import datasets
from datasets import load_metric

import argparse
import transformers
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    default_data_collator,
)
from quant_transformer.quantization.state import (
    enable_calibration_woquantization,
    enable_quantization,
    disable_all,
    set_observer_name,
)
import quant_transformer.solver.utils.qa_utils as qa_utils
from quant_transformer.solver.utils.qa_utils import postprocess_qa_predictions, prepare_validation_features
from quant_transformer.solver.utils.qa_utils import QuestionAnsweringTrainer
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


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def evaluate(trainer, eval_datasets, eval_examples):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=eval_datasets, eval_examples=eval_examples)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def prepare_input_output(trainer, cali_data):

    logger.info('**prepare fp input and output**')
    disable_all(trainer.model)
    data_loader = trainer.get_eval_dataloader(cali_data)
    fp_input, fp_output = [], []
    with torch.no_grad():
        for p in data_loader:
            tmp = {}
            for k, v in p.items():
                tmp[k] = v.cuda()
            outputs = trainer.model(**tmp)
            output = []
            attention_mask = tmp['attention_mask']
            start_logits = outputs[0][attention_mask == 1].detach()
            end_logits = outputs[1][attention_mask == 1].detach()
            output.append(start_logits)
            output.append(end_logits)
            fp_input.append(tmp)
            fp_output.append(output)
    return fp_input, fp_output


def calibrate(trainer, fp_input):
    logger.info("*** Calibrate ***")
    with torch.no_grad():
        for batch in fp_input:
            trainer.model(**batch)


def main(config_path):
    config = qa_utils.parse_config(config_path)
    set_seed(config.train.seed)
    training_args = make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    raw_datasets = qa_utils.load_qa_dataset(config.data)
    tokenizer, model = qa_utils.load_model(config.model)

    if config.data.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({config.data.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    config.data.max_seq_length = min(config.data.max_seq_length, tokenizer.model_max_length)
    model.cuda()
    model.eval()
    if getattr(config, 'quant', None):
        model = quantize_model(model, config)
    train_examples = qa_utils.check_return_data(raw_datasets, "train", True, config.data.max_train_samples)
    eval_examples = qa_utils.check_return_data(raw_datasets, "validation", True, config.data.max_eval_samples)
    column_names = eval_examples.column_names
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_datasets = eval_examples.map(
            prepare_validation_features,
            fn_kwargs={'tokenizer': tokenizer, 'config_data': config.data, 'column_names': column_names},
            batched=True,
            num_proc=config.data.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.data.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        if config.data.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_datasets), config.data.max_eval_samples)
            eval_datasets = eval_datasets.select(range(max_eval_samples))
    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if config.data.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=config.data.version_2_with_negative,
            n_best_size=config.data.n_best_size,
            max_answer_length=config.data.max_answer_length,
            null_score_diff_threshold=config.data.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=config.progress.log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if config.data.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    logger.info(f"load metric version_2_with_negative: {config.data.version_2_with_negative}")
    metric = load_metric(
        "squad_v2" if config.data.version_2_with_negative else "squad",
        cache_dir=config.data.cache_dir
    )

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize the Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        eval_examples=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    if getattr(config, 'quant', None):
        # calibrate the model first
        calibrate_examples = train_examples.shuffle().select(range(config.quant.calibrate))
        assert tokenizer.padding_side == "right"
        with training_args.main_process_first(desc="Calibrate dataset map pre-processing"):
            cali_data = calibrate_examples.map(
                prepare_validation_features,
                fn_kwargs={'tokenizer': tokenizer, 'config_data': config.data, 'column_names': column_names},
                batched=True,
                num_proc=config.data.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not config.data.overwrite_cache,
                desc="Running tokenizer on calibrate dataset",
            )
        fp_input, fp_output = prepare_input_output(trainer, cali_data)
        if config.quant.ln.delay:
            from gamma_migration import delay_ln
            trainer.model = delay_ln(trainer.model, config.quant, config.model)
        torch.cuda.empty_cache()
        # calibrate the weight
        enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
        calibrate(trainer, [fp_input[0]])
        if 'PruneMinMaxObserver' in config.quant.a_qconfig.observer:
            disable_all(trainer.model)
            set_observer_name(trainer.model)
            token_wise_clipping.token_wise_clipping(trainer, fp_input, fp_output, config)
            if 'LSQ' in config.quant.a_qconfig.quantizer:
                # OOM, use smaller batch size for learning
                trainer.args.per_device_eval_batch_size = 8
                fp_input, fp_output = prepare_input_output(trainer, cali_data)
                token_wise_clipping.learn_scale(trainer, fp_input, fp_output,
                                                getattr(config.quant, 'learn', {'lr': 1e-5, 'epoch': 3}))
                trainer.args.per_device_eval_batch_size = 32
        else:
            # calibrate the activation
            enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
            calibrate(trainer, fp_input)

        torch.cuda.empty_cache()
    if training_args.do_eval:
        if getattr(config, 'quant', None):
            enable_quantization(trainer.model)
        evaluate(trainer, eval_datasets, eval_examples)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # general parameters for data and model
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
