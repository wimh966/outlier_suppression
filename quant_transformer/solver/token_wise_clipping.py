from torch.nn import MSELoss
import torch
import logging
from quant_transformer.quantization.fake_quant import QuantizeBase, LSQPlusFakeQuantize, LSQFakeQuantize
from quant_transformer.quantization.state import disable_all
logger = logging.getLogger("transformer")
# support ptq glue, ptq squad, ptq summ
task_type = None
model_type = None


def set_ratio(model, ratio):
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if 'act' in name:
                module.observer.set_percentile(ratio)
                module.observer.cnt = 0
                module.disable_fake_quant()
                module.enable_observer()


def enable_quantization(model):
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase) and 'act' in name:
            submodule.disable_observer()
            submodule.enable_fake_quant()


def calibrate(model, fp_input, fp_output=None):
    loss = 0
    with torch.no_grad():
        for i, batch in enumerate(fp_input):
            outputs = model(**batch)
            if fp_output is not None:
                if task_type == 'glue':
                    loss += loss_fct(outputs[0], fp_output[i])
                elif task_type in ('squad', 'squad_v2'):
                    start_logits = outputs[0][batch['attention_mask'] == 1]
                    end_logits = outputs[1][batch['attention_mask'] == 1]
                    loss += loss_fct(start_logits, fp_output[i][0])
                    loss += loss_fct(end_logits, fp_output[i][1])
                elif task_type == 'summ':
                    output = outputs[0][batch['decoder_attention_mask'] == 1, :]
                    loss += loss_fct(output, fp_output[i])
                else:
                    raise NotImplementedError
    return loss


def find_ratio(trainer, fp_input, fp_output, param):
    p, loss = 0, 10000000
    iters = param['iters']
    step = param['step']
    for i in range(iters):
        set_ratio(trainer.model, 1.0 - step * i)
        calibrate(trainer.model, fp_input)
        enable_quantization(trainer.model)
        cur_loss = calibrate(trainer.model, fp_input, fp_output)
        logger.info('the ratio is {}, the loss is {}'.format(1.0 - step * i, cur_loss))
        if loss > cur_loss:
            loss = cur_loss
            p = i
    ratio = 1.0 - step * p
    logger.info('the best percentile is {}'.format(ratio))
    set_ratio(trainer.model, ratio)
    calibrate(trainer.model, fp_input)


loss_fct = MSELoss()


def learn_scale(trainer, fp_input, fp_output, config_quant_learn):
    disable_all(trainer.model)
    logger.info('*** begin learn the scale now! ***')
    torch.cuda.empty_cache()
    para = []
    for name, module in trainer.model.named_modules():
        if isinstance(module, QuantizeBase) and 'act' in name:
            module.enable_fake_quant()
            module.disable_observer()
            if isinstance(module, LSQPlusFakeQuantize):
                para.append(module.scale)
                para.append(module.zero_point)
            elif isinstance(module, LSQFakeQuantize):
                para.append(module.scale)
    opt = torch.optim.Adam(para, lr=config_quant_learn['lr'])
    iters = config_quant_learn['epoch'] * len(fp_input)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.)
    for j in range(config_quant_learn['epoch']):
        for i, batch in enumerate(fp_input):
            opt.zero_grad()
            outputs = trainer.model(**batch)
            if task_type == 'glue':
                loss = loss_fct(outputs[0], fp_output[i])
            elif task_type in ('squad', 'squad_v2'):
                start_logits = outputs[0][batch['attention_mask'] == 1]
                end_logits = outputs[1][batch['attention_mask'] == 1]
                loss = loss_fct(start_logits, fp_output[i][0])
                loss += loss_fct(end_logits, fp_output[i][1])
            elif task_type == 'summ':
                output = outputs[0][batch['decoder_attention_mask'] == 1, :]
                loss = loss_fct(output, fp_output[i])
            else:
                raise NotImplementedError
            loss.backward()
            opt.step()
            scheduler.step()
    torch.cuda.empty_cache()


a_bit_iters = {
    8: 0.1,
    6: 0.3,
    4: 0.9
}


def cac_step_iters(a_bit, bs, config_data):
    # can calculate the step based on seq_length and batch_size
    if hasattr(config_data, 'max_seq_length'):
        seq_token = config_data.max_seq_length
    else:
        seq_token = config_data.max_source_length
    step = 128 * 32 * 0.01 / bs / seq_token
    step = float(format(step, '.2g'))
    step = min(step, 0.01)
    iters = int(a_bit_iters[a_bit] / step)
    logger.info('the step is {}, the iters is {}'.format(step, iters))
    return step, iters


def token_wise_clipping(trainer, fp_input, fp_output, config):
    global model_type
    global task_type
    model_type = config.model.model_type
    task_type = config.model.task_type
    config_quant = config.quant

    logger.info("*** Evaluate Token Percentile ***")
    step, iters = cac_step_iters(config_quant.a_qconfig.bit,
                                 trainer.args.per_device_eval_batch_size,
                                 config.data)

    find_ratio(trainer, fp_input, fp_output,
               {'iters': getattr(config.quant, 'iters', iters),
                'step': getattr(config.quant, 'step', step)})
