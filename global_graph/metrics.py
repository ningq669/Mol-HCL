import torch
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Union
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from utils import gen_preds, eval_threshold

def get_roc_score(model,
                  features, 
                  adj: torch.sparse.FloatTensor,
                  adj_tensor, 
                  drug_nums,
                  edges_pos: np.ndarray, edges_neg: Union[np.ndarray, List[list]], test = None) -> Tuple[float, float]:
    model.eval()
    
    rec, emb = model(features, adj, adj_tensor, drug_nums, return_embeddings = True)


    emb = emb.detach().cpu().numpy()#(544,4096)
    rec = rec.detach().cpu().numpy()#(544,544)
    adj_rec = rec
    print(adj_rec.shape)

    preds, preds_neg = gen_preds(edges_pos, edges_neg, adj_rec)
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    preds_all, preds_all_ = eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test)

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    f1_score_ = f1_score(labels_all, preds_all_)
    acc_score = accuracy_score(labels_all, preds_all_)
    return roc_score, ap_score, f1_score_, acc_score


# def get_roc_score2(model,
#                   features,
#                   adj: torch.sparse.FloatTensor,
#                   adj_tensor,
#                   drug_nums,
#                   edges_pos: np.ndarray, edges_neg: Union[np.ndarray, List[list]], test=None) -> Tuple[float, float]:
#     model.eval()
#
#     rec, emb = model(features, adj, adj_tensor, drug_nums, return_embeddings=True)
#
#     emb = emb.detach().cpu().numpy()  # (544,4096)
#     rec = rec.detach().cpu().numpy()  # (544,544)
#     adj_rec = rec
#
#     output_file = 'drug_predictions.xlsx'
#     preds, preds_neg = gen_preds2(edges_pos, edges_neg, adj_rec)
#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
#     preds_all, preds_all_ = eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test)
#
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)
#     f1_score_ = f1_score(labels_all, preds_all_)
#     acc_score = accuracy_score(labels_all, preds_all_)
#     return roc_score, ap_score, f1_score_, acc_score

