import os
import numpy as np
import logging
import sys
import nltk
import argparse
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import datasets
import random
from tqdm import tqdm
from datasets import load_metric
import torch  # noqa E401
from torch.utils.data import DataLoader

import quant_transformer.solver.utils.summ_utils as summ_utils
from quant_transformer.quantization.state import enable_calibration_woquantization, enable_quantization, \
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
    training_args = Seq2SeqTrainingArguments(
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
        label_smoothing_factor=config_train.label_smoothing_factor,
        gradient_checkpointing=config_train.gradient_checkpointing,
        generation_max_length=config_train.generation_max_length,
        predict_with_generate=config_train.predict_with_generate,
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


def evaluate(trainer, eval_dataset, training_args, config_data):
    logger.info("*** Evaluate ***")
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else config_data.val_max_target_length
    )
    num_beams = config_data.num_beams if config_data.num_beams is not None else training_args.generation_num_beams
    metrics = trainer.evaluate(eval_dataset=eval_dataset,
                               max_length=max_length,
                               num_beams=num_beams,
                               metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def prepare_input_output(trainer, cali_data, training_args, config_data):
    logger.info('**prepare fp input and output**')
    disable_all(trainer.model)
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else config_data.val_max_target_length
    )
    num_beams = config_data.num_beams if config_data.num_beams is not None else training_args.generation_num_beams
    data_loader = trainer.get_eval_dataloader(cali_data)
    fp_input, fp_output = [], []
    with torch.no_grad():
        for batch in data_loader:
            generated_tokens = trainer.model.generate(
                batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                max_length=max_length,
                num_beams=num_beams,
            )
            decoder_attention_mask = torch.zeros_like(generated_tokens)
            decoder_attention_mask[generated_tokens != trainer.model.config.pad_token_id] = 1
            tmp = {}
            tmp['input_ids'] = batch['input_ids'].cuda()
            tmp['attention_mask'] = batch['attention_mask'].cuda()
            tmp['decoder_input_ids'] = generated_tokens.cuda()
            tmp['decoder_attention_mask'] = decoder_attention_mask.cuda()
            fp_input.append(tmp)
            output = trainer.model(**tmp)[0]
            output = output[decoder_attention_mask == 1, :].detach()
            fp_output.append(output)
    return fp_input, fp_output


def calibrate(trainer, fp_input):
    logger.info("*** Calibrate ***")
    with torch.no_grad():
        for batch in fp_input:
            trainer.model(**batch)


def main(config_path):
    config = summ_utils.parse_config(config_path)
    set_seed(config.train.seed)
    training_args = make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    raw_datasets = summ_utils.load_summ_dataset(config.model, config.data)
    tokenizer, model = summ_utils.load_model(config.model, config.data)

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
    model.cuda()
    model.eval()
    if getattr(config, 'quant', None):
        fp_model = model
        fp_model.eval()
        model = quantize_model(model, config)
    train_dataset = summ_utils.check_return_data(raw_datasets, "train", True, config.data.max_train_samples)
    eval_dataset = summ_utils.check_return_data(raw_datasets, "validation", True, config.data.max_eval_samples)
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            summ_utils.preprocess_function,
            fn_kwargs={'tokenizer': tokenizer,
                       'config_data': config.data,
                       'max_target_length': config.data.val_max_target_length},
            batched=True,
            num_proc=config.data.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not config.data.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    label_pad_token_id = -100 if config.data.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if config.data.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    if getattr(config, 'quant', None):
        # calibrate the model
        calibrate_dataset = train_dataset.shuffle().select(range(config.quant.calibrate))
        with training_args.main_process_first(desc="calibration dataset map pre-processing"):
            cali_data = calibrate_dataset.map(
                summ_utils.preprocess_function,
                fn_kwargs={'tokenizer': tokenizer,
                           'config_data': config.data,
                           'max_target_length': config.data.val_max_target_length},
                batched=True,
                num_proc=config.data.preprocessing_num_workers,
                remove_columns=calibrate_dataset.column_names,
                load_from_cache_file=not config.data.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        fp_input, fp_output = prepare_input_output(trainer, cali_data, training_args, config.data)
        if config.quant.ln.delay:
            from gamma_migration import delay_ln
            trainer.model = delay_ln(trainer.model, config.quant, config.model)

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
        evaluate(trainer, eval_dataset, training_args, config.data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # general parameters for data and model
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
