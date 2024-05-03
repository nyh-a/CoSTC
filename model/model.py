import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import numpy as np
from itertools import chain 
from sentence_transformers.cross_encoder import CrossEncoder
import scipy.sparse as smat

from base import BaseModel
from utils import mean_pooling
from model.modules import *
from model.loss import *


class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class Model_Prompt(nn.Module):
    """
    using [MASK] hidden state as sentence embeddding 
    """
    def __init__(self, model_name, normalize=True):
        super().__init__()
        # encoder 
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.normalize = normalize

    def forward(self, **kwargs):
        model_output = self.model(**kwargs,  return_dict=True)
        last_hidden = model_output.last_hidden_state
        m_mask = kwargs['input_ids'] == self.tokenizer.mask_token_id 
        embeddings = last_hidden[m_mask]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
        
