"""custom metrics"""
import numpy as np
from torch import nn


class MovingAverage(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_loss(predictions, targets):
    """compute loss given predictions and targets

    Given a tuple of predicted arrays and their corresponding target arrays,
    return all losses

    Args:
        predictions: contains predictions for start_position, end_position and
            class labels
        targets: contains target arrays for start position, end position and
            class labels

    Returns:
        start_position_loss: loss for start position
        end_position_loss: loss for end position
        classifier_loss: loss for class labels
    """
    start_position_preds, end_position_preds, classifier_preds = predictions
    start_position_labels, end_position_labels, classifier_labels = targets

    start_position_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_position_preds, start_position_labels)
    end_position_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_position_preds, end_position_labels)
    classifier_loss = nn.CrossEntropyLoss()(classifier_preds, classifier_labels)
    return start_position_loss, end_position_loss, classifier_loss


def get_position_accuracy(logits, labels):
    predictions = np.argmax(nn.functional.softmax(logits, dim=1).cpu().data.numpy(), axis=1)
    total_num = 0
    sum_correct = 0
    for i in range(len(labels)):
        if labels[i] >= 0:
            total_num += 1
            if predictions[i] == labels[i]:
                sum_correct += 1
    if total_num == 0:
        total_num = 1e-7
    return np.float32(sum_correct) / total_num, total_num


def compute_accuracy(predictions, targets):
    """Given predictions and targets, compute accuracy

    Args:
        predictions: contains predictions for start_position, end_position and
            class labels
        targets: contains target arrays for start position, end position and
            class labels

    Returns:
        all accuracies and all counts

    """
    start_position_logits, end_position_logits, classifier_logits = predictions
    start_position_labels, end_position_labels, classifier_labels = targets

    start_acc, start_position_num = get_position_accuracy(start_position_logits, start_position_labels)
    end_acc, end_position_num = get_position_accuracy(end_position_logits, end_position_labels)
    class_acc, class_num = get_position_accuracy(classifier_logits, classifier_labels)

    return start_acc, end_acc, class_acc, start_position_num, end_position_num, class_num
