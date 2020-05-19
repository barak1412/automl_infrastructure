from automl_infrastructure.experiment.metrics import MetricFactory, Metric


def parse_metric(metric):
    if isinstance(metric, str):
        metric_obj = MetricFactory.create(metric)
        metric_func = lambda y_true, classifier_prediction: metric_obj.measure(y_true, classifier_prediction)
    elif isinstance(metric, Metric):
        metric_func = lambda y_true, classifier_prediction: metric.measure(y_true, classifier_prediction)
    elif callable(metric):
        metric_func = metric
    else:
        raise Exception('Unsupported given metric (scoring).')
    return metric_func
