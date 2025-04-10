from typing import Union, Tuple
from .encoder import GAT, TopoGAT
from .decoder import InnerProductDecoder
from features import BatchMolGraph
from .deepinfomax import GcnInfomax
from models.model import *
from global_graph.pkl import load_pkl_as_tensor



class Config:
    def __init__(self):
        self.datapath = './datasets'
        self.kfold = 5
        self.batchSize = 128
        self.ratio = 0.2
        self.epoch = 8
        self.gcn_layers = 2
        self.view = 3
        self.inSize = 4096
        self.outSize = 4096
        self.nodeNum =32
        self.nodeNum2=1
        self.hdnDropout = 0.7
        self.fcDropout = 0.5
        self.maskMDI = False
        self.device = torch.device('cuda')

param=Config()


class CrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim):
        super(CrossAttention, self).__init__()

        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)

    def forward(self, input_a, input_b):
        mapped_a = self.linear_a(input_a)
        mapped_b = self.linear_b(input_b)
        y = mapped_b.transpose(0, 1)

        scores = torch.matmul(mapped_a, mapped_b.transpose(0, 1))
        attentions_a = torch.softmax(scores, dim=-1)
        attentions_b = torch.softmax(scores.transpose(0, 1),
                                     dim=-1)
        output_a = torch.matmul(attentions_b, input_b)
        output_b = torch.matmul(attentions_a.transpose(0, 1), input_a)

        return output_a,output_b


class HTCL(nn.Module):

    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.3, bias: bool = False,
                 sparse: bool = True):
        super(HTCL, self).__init__()
        self.num_features = num_features
        self.features_nonzero = features_nonzero
        self.dropout = dropout
        self.bias = bias
        self.sparse = sparse
        self.args = args
        self.create_encoder(args)       

        self.struc_enc = self.select_encoder1(args)
        self.seman_enc = self.select_encoder2(args)
        self.dec_local = InnerProductDecoder(args.hidden_size)
        self.dec_global = InnerProductDecoder(args.hidden_size) 
        self.sigmoid = nn.Sigmoid()
        self.DGI_setup()
        self.md_supernode = MDI(param)
        self.md_supernode2 = MDI2(param)
        self.create_ffn(args)
        self.hidden_size=args.hidden_size

    def create_encoder(self, args: Namespace, vocab: Vocab = None):
        self.encoder = MPN(args)
        return self.encoder


    def select_encoder1(self, args: Namespace):
        return GAT(args, self.num_features + self.args.input_features_size,
                                          self.features_nonzero,
                                          dropout=self.dropout, bias=self.bias,
                                          sparse=self.sparse)


    def select_encoder2(self, args: Namespace):
        return TopoGAT(args, self.num_features + self.args.input_features_size,
                                          self.features_nonzero,
                                       dropout=self.dropout, bias=self.bias,
                                          sparse=self.sparse)


    def create_ffn(self, args: Namespace):
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        self.fusion_ffn_local = nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.fusion_ffn_global = nn.Linear(args.gat_hidden*8, args.ffn_hidden_size)

        ffn = []
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(args.ffn_hidden_size, args.drug_nums),
        ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.dropout = dropout


    def DGI_setup(self):
        self.DGI_model1 = GcnInfomax(self.args)
        self.DGI_model2 = GcnInfomax(self.args)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                adj: torch.sparse.FloatTensor,
                adj_tensor,
                drug_nums,
                return_embeddings: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        smiles_batch = batch #544
        features_batch = None

        # molecular view
        feat_orig = self.encoder(smiles_batch, features_batch)
        feat = self.dropout(feat_orig)
        fused_feat = self.fusion_ffn_local(feat)
        output = self.ffn(fused_feat)
        outputs = self.sigmoid(output)
        outputs_l = outputs.view(-1)

        # structural view
        embeddings1 = self.struc_enc(feat_orig, adj)
        outputs_1 = self.md_supernode2(embeddings1, embeddings1)
        outputs_1 = outputs_1 * 0.5 + embeddings1 * 0.5
        feat_g1 = self.dropout(outputs_1)
        fused_feat_g1 = self.fusion_ffn_global(feat_g1)
        output_g1 = self.ffn(fused_feat_g1)
        outputs_6 = self.sigmoid(output_g1)
        outputs_g1 = outputs_6.view(-1)

        # semantic view
        embeddings2 = self.seman_enc(feat_orig, adj)
        outputs_2 = self.md_supernode(embeddings2, embeddings2)
        outputs_2 = outputs_2 * 0.6 + embeddings2 * 0.4
        feat_g2 = self.dropout(outputs_2)
        fused_feat_g2 = self.fusion_ffn_global(feat_g2)
        output_g2 = self.ffn(fused_feat_g2)
        outputs_7 = self.sigmoid(output_g2)
        outputs_g2 = outputs_7.view(-1)

        # intergrate the two views
        cross_attention = CrossAttention(outputs_1.shape[-1], outputs_2.shape[-1],self.hidden_size)
        cross_attention=cross_attention.cuda()
        output_a, output_b = cross_attention(outputs_1, outputs_2)

        embeddings = output_a+output_b
        feat_g = self.dropout(embeddings)
        fused_feat_g = self.fusion_ffn_global(feat_g)
        output_g = self.ffn(fused_feat_g)
        outputs = self.sigmoid(output_g)

        local_embed = feat_orig

        DGI_loss1 = self.DGI_model1(outputs_1, local_embed, adj_tensor, drug_nums)
        DGI_loss2 = self.DGI_model2(outputs_2, local_embed, adj_tensor, drug_nums)


        if return_embeddings:
            return outputs, embeddings

        return outputs_l, outputs_g1, outputs_g2,DGI_loss1, DGI_loss2
