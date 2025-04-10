from argparse import Namespace
import numpy as np
import torch.nn as nn

from typing import List
from .mpn import MPN
from data.mol_tree import Vocab
from nn_utils import get_activation_function, initialize_weights
import torch
from otherlayers import *
import pandas as pd

class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network followed by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace, vocab: Vocab = None):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim
        if args.pooling == 'lstm':
            first_linear_dim *= (1 * 2)

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)
        return output


FEATURES_FUSIONER_REGISTRY = {}


def register_features_fusioner(features_fusioner_name: str):
    def decorator(features_fusioner):
        FEATURES_FUSIONER_REGISTRY[features_fusioner_name] = features_fusioner
        return features_fusioner

    return decorator


def get_features_fusioner(features_fusioner_name):
    if features_fusioner_name not in FEATURES_FUSIONER_REGISTRY:
        raise ValueError(f'Features fusioner "{features_fusioner_name}" could not be found. '
                         f'If this fusioner relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_FUSIONER_REGISTRY[features_fusioner_name]


def get_available_features_fusioners():
    """Returns the names of available features generators."""
    return list(FEATURES_FUSIONER_REGISTRY.keys())


def build_model(args: Namespace, ddi:bool = False) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes
    if args.dataset_type == 'multilabel':
        args.output_size = args.num_labels

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')

    if args.jt and args.jt_vocab_file is not None:
        vocab = [x.strip("\r\n ") for x in open(args.jt_vocab_file, 'r')]
        vocab = Vocab(vocab)
    else:
        vocab = None
    model.create_encoder(args, vocab=vocab)
    model.create_ffn(args)

    initialize_weights(model)
    return model

class MDI2(nn.Module):
    def __init__(self, param):
        super(MDI2, self).__init__()

        self.inSize = param.inSize
        self.outSize = param.outSize
        self.gcnlayers = param.gcn_layers
        self.device = param.device
        self.nodeNum = param.nodeNum2
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.maskMDI = param.maskMDI
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()

        self.nodeEmbedding = BnodeEmbedding(
            torch.tensor(np.random.normal(size=(max(self.nodeNum, 0), self.inSize)), dtype=torch.float32),
            dropout=self.hdnDropout).to(self.device)

        self.nodeGCN = GCN(self.inSize, self.outSize, dropout=self.hdnDropout, layers=self.gcnlayers, resnet=True,
                           actFunc=self.relu1).to(self.device)

        self.fcLinear = MLP(self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)

        self.layeratt_m = LayerAtt(self.inSize, self.outSize, self.gcnlayers)
        self.layeratt_d = LayerAtt(self.inSize, self.outSize, self.gcnlayers)

    def forward(self, em, ed):
        xm = em.unsqueeze(1)
        xd = ed.unsqueeze(1)

        conv1d = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1).to(xm.device)
        conv2d = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1).to(xm.device)
        conv3d = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=1).to(xm.device)

        xm1 = conv1d(xm)
        xd1 = conv1d(xd)
        xm2 = conv2d(xm1)
        xd2 = conv2d(xd1)
        xm = conv3d(xm2)
        xd= conv3d(xd2)

        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(xd), 1, 1)
            node = torch.cat([xm, xd,node], dim=1)
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)
            cosNode = torch.matmul(node, node.transpose(1, 2)) / (nodeDist * nodeDist.transpose(1, 2) + 1e-8)
            cosNode = self.relu2(cosNode)

            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            if self.maskMDI: cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0

            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(len(xm), 1, 1)
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5)
            pL = torch.matmul(torch.matmul(D, cosNode), D)
            mGCNem, dGCNem = self.nodeGCN(node, pL)
            mLAem = self.layeratt_m(mGCNem)
            dLAem = self.layeratt_d(dGCNem)
            node_embed = mLAem * dLAem

        return node_embed

class MDI(nn.Module):
    def __init__(self, param):
        super(MDI, self).__init__()

        self.inSize = param.inSize
        self.outSize = param.outSize
        self.gcnlayers = param.gcn_layers
        self.device = param.device
        self.nodeNum = param.nodeNum
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.maskMDI = param.maskMDI
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()

        self.nodeEmbedding = BnodeEmbedding(
            torch.tensor(np.random.normal(size=(max(self.nodeNum, 0), self.inSize)), dtype=torch.float32),
            dropout=self.hdnDropout).to(self.device)

        self.nodeGCN = GCN(self.inSize, self.outSize, dropout=self.hdnDropout, layers=self.gcnlayers, resnet=True,
                           actFunc=self.relu1).to(self.device)

        self.fcLinear = MLP(self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)

        self.layeratt_m = LayerAtt(self.inSize, self.outSize, self.gcnlayers)
        self.layeratt_d = LayerAtt(self.inSize, self.outSize, self.gcnlayers)

    def forward(self, em, ed):
        xm = em.unsqueeze(1)
        xd = ed.unsqueeze(1)

        conv1d = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1).to(xm.device)
        conv2d = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=1).to(xm.device)
        conv3d = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1).to(xm.device)

        xm1 = conv1d(xm)
        xd1 = conv1d(xd)
        xm2 = conv2d(xm1)
        xd2 = conv2d(xd1)
        xm = conv3d(xm2)
        xd= conv3d(xd2)

        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(xd), 1, 1)
            node = torch.cat([xm, xd,node], dim=1)
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)
            cosNode = torch.matmul(node, node.transpose(1, 2)) / (nodeDist * nodeDist.transpose(1, 2) + 1e-8)
            cosNode = self.relu2(cosNode)

            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            if self.maskMDI: cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0


            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(len(xm), 1, 1)
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5)
            pL = torch.matmul(torch.matmul(D, cosNode), D)

            mGCNem, dGCNem = self.nodeGCN(node, pL)
            mLAem = self.layeratt_m(mGCNem)
            dLAem = self.layeratt_d(dGCNem)
            node_embed = mLAem * dLAem


        return node_embed

