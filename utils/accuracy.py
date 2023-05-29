import torch

def calculate_topk_accuracy(preds, labels, topk=(1, 5)):
    """ Calculate the top-k accuracy given predictions and labels """
    maxk = max(topk)
    batch_size = labels.size(0)

    _, pred_indices = preds.topk(maxk, 1, True, True)
    pred_indices = pred_indices.t()
    
    correct = pred_indices.eq(labels.view(1, -1).expand_as(pred_indices))

    topk_acc = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        acc = correct_k * (100.0 / batch_size)
        topk_acc.append(acc)
    
    return topk_acc
