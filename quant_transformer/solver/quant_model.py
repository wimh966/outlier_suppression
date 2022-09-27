import copy
from quant_transformer.model.quant_bart import QuantizedBartForConditionalGeneration, \
    QuantizedBartForSequenceClassification, QuantizedBartForQuestionAnswering  # noqa: F401
from quant_transformer.model.quant_bert import QuantizedBertForSequenceClassification, \
    QuantizedBertForQuestionAnswering  # noqa: F401
from quant_transformer.model.quant_roberta import QuantizedRobertaForSequenceClassification, \
    QuantizedRobertaForQuestionAnswering  # noqa: F401
_SUPPORT_MODELS = ['bert', 'roberta', 'bart']


def get_model_task_type(model_name, config_data):
    if config_data.dataset_name in ('cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'sst2', 'rte', 'stsb'):
        task_type = 'glue'
    elif config_data.dataset_name in ('squad', 'squad_v2'):
        task_type = config_data.dataset_name
    elif config_data.dataset_name in ('cnn_dailymail', 'xsum'):
        task_type = 'summ'
    else:
        raise NotImplementedError
    if 'roberta' in model_name:
        model_type = 'roberta'
    elif 'bert' in model_name:
        model_type = 'bert'
    elif 'bart' in model_name:
        model_type = 'bart'
    else:
        raise NotImplementedError
    return task_type, model_type


def quantize_model(fp_model, config):
    config_quant = config.quant
    config_model = config.model
    config_quant.backend = config_quant.get('backend', 'academic')
    config_quant.is_remove_padding = config_quant.get('is_remove_padding', True)
    config_quant.ln = config_quant.get('ln', dict())
    config_quant.ln.delay = config_quant.ln.get('delay', False)

    model_name = fp_model.__class__.__name__.lower()
    task_type, model_type = get_model_task_type(model_name, config.data)
    config_model.model_type = model_type
    config_model.task_type = task_type
    fp_model.eval()
    model = copy.deepcopy(fp_model)
    model = eval("Quantized" + str(fp_model.__class__.__name__))(
        model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=False,
        backend=config_quant.backend, is_remove_padding=config_quant.is_remove_padding,
    )
    model.eval()
    return model
