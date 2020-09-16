import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import mixup_process


class Text_CNN(nn.Module):

    def __init__(self, num_classes, dropout, scales, tokenizer):
        super(Text_CNN, self).__init__()

        WN = len(tokenizer.vocab)
        WD = 300
        KN = 128
        KS = [3, 4, 5]
        C = num_classes

        self.word_embedding = nn.Embedding(WN, WD, padding_idx=0)

        self.conv_layer = nn.ModuleList([nn.Conv1d(WD, KN, K, padding=K//2, bias=True) for K in KS])
        self.conv_layer_mask = nn.ModuleList([nn.Conv1d(1, KN, K, padding=K//2, bias=False) for K in KS])
        for conv_mask in self.conv_layer_mask:
            conv_mask.weight.data.fill_(1.)
            conv_mask.weight.requires_grad_(False)

        self.linear = nn.Linear(len(KS) * KN, C)
        self.dropout = nn.Dropout(dropout)

        self.original_params = nn.ParameterDict()
        self.perturb_params = nn.ParameterDict()
        self.perturb_modules = nn.ModuleDict({
            'word_emb': self.word_embedding,
            'conv': self.conv_layer,
            'linear': self.linear
        })
        self.perturb_scale = {
            'word_emb': scales[0],
            'conv': scales[1],
            'linear': scales[2]
        }

    def forward(self, text, adv=False, perturbation=None, lamda=None, indices=None):
        embedding = self.word_embedding(text)
        if adv:
            embedding.requires_grad_(True)
        if lamda is not None:
            embedding = mixup_process(embedding, lamda, indices)
        if perturbation is not None:
            embedding += perturbation
        embedding = self.dropout(embedding)
        mask_input = (text!=0).float().unsqueeze(-1)
        cnn_out = [torch.relu(conv(embedding.transpose(1, 2))) for conv in self.conv_layer]
        mask_out = [conv(mask_input.transpose(1, 2)) for conv in self.conv_layer_mask]
        cnn_out = [torch.where(mask_out[i]!=0, cnn_out[i], torch.zeros_like(cnn_out[i])) for i in range(len(cnn_out))]
        output = torch.cat([F.max_pool1d(out, out.size(-1)).squeeze(-1) for out in cnn_out], dim=-1)
        output = self.linear(self.dropout(output))
        if adv:
            return output, embedding
        else:
            return output


def textcnn(num_classes, dropout, scales, tokenizer):
    return Text_CNN(num_classes, dropout, scales, tokenizer)
