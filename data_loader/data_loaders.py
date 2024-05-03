from functools import partial
from itertools import chain, product

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers.cross_encoder import CrossEncoder

from .dataset import *


class DataLoader_Prompt_Random(DataLoader):
    """
    Random negetive sampling
    Convert to input_ids for Language Model
    """
    def __init__(self, data_path, taxonomy_name, sampling_mode=0, batch_size=16, negative_size=32, shuffle=True, num_workers=8, 
                 tokenizer='/codes/share/huggingface_models/bert_base_uncased'):
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.negative_size = negative_size

        raw_graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=False, existing_partition=True)
        self.dataset = Dataset_random2(graph_dataset=raw_graph_dataset, mode="train", 
                                    sampling_mode=sampling_mode, negative_size=negative_size)
        
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer)

        self.template = "The meaning of {} is [MASK]"
        
        self.num_workers = num_workers
        super(DataLoader_Prompt_Random, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=shuffle,
                                                collate_fn=self.collate_fn,
                                                num_workers=self.num_workers, pin_memory=False)
        
    def collate_fn(self, samples):
        # input: query_node_id--<batch_size>     candidate_position--<batch_size, negative_size, 3> (p,c,s)
        #        candidate_labels--<batch_size, negative_size, 3> (p,c,pc)
        
        query_node_id,   p_id, c_id, s_id,     p_label, c_label, pc_label  = map(list, zip(*chain(*samples)))
        
        # convert to desc's input_id    
        query_node_id = torch.tensor(query_node_id)
        target = torch.tensor(pc_label, dtype=torch.long)

        desc_p = self.dataset.id2desc[torch.tensor(p_id)]       # batch_size * negative_size
        desc_c = self.dataset.id2desc[torch.tensor(c_id)]
        desc_s = self.dataset.id2desc[torch.tensor(s_id)]
        desc_q = self.dataset.id2desc[query_node_id]

        sentences_query = []
        sentences_pos = []
        for idx in range(query_node_id.shape[0]):       # for each sample
            # query
            sentences_query.append(f"The meaning of \"{desc_q[idx]}\" is [MASK]")
            # position
            desc = desc_p[idx] + '\n' + desc_c[idx] + '\n' + desc_s[idx]
            sentences_pos.append(f"The meaning of \"{desc}\" is [MASK]")

            # # query
            # sentences_query.append(f"This sentence : \"{desc_q[idx]}\" means [MASK].")
            # # position
            # desc = desc_p[idx] + '\n' + desc_c[idx] + '\n' + desc_s[idx]
            # sentences_pos.append(f"This sentence : \"{desc}\" means [MASK].")


        encoded_query = self.tokenizer(sentences_query, padding=True, truncation=True, return_tensors='pt')
        encoded_pos = self.tokenizer(sentences_pos, padding=True, truncation=True, return_tensors='pt')
    
        return encoded_query, encoded_pos, target

    def __str__(self):
        return "\n\t".join([
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
        ])
    


class DataLoader_Prompt_NS(DataLoader):
    """
    Random negetive sampling
    Convert to input_ids for Language Model
    """
    def __init__(self, data_path, taxonomy_name, sampling_mode=0, batch_size=16, negative_size=32, shuffle=True, num_workers=8, 
                 tokenizer='/codes/share/huggingface_models/bert_base_uncased', fixed_sibling=False):
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.negative_size = negative_size

        raw_graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=False, existing_partition=True)
        self.dataset = Dataset_Diversity(graph_dataset=raw_graph_dataset, mode="train", 
                                    sampling_mode=sampling_mode, negative_size=negative_size, fixed=fixed_sibling)
        
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer)

        self.template = "The meaning of {} is [MASK]"

        
        self.num_workers = num_workers
        super(DataLoader_Prompt_NS, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=shuffle,
                                                collate_fn=self.collate_fn,
                                                num_workers=self.num_workers, pin_memory=False)
        
    def collate_fn(self, samples):
        # input: query_node_id--<batch_size>     candidate_position--<batch_size, negative_size, 3> (p,c,s)
        #        candidate_labels--<batch_size, negative_size, 3> (p,c,pc)
        
        query_node_id,   p_id, c_id, s_id,     p_label, c_label, pc_label  = map(list, zip(*chain(*samples)))
        
        # convert to desc's input_id    
        query_node_id = torch.tensor(query_node_id)
        target = torch.tensor(pc_label, dtype=torch.long)

        desc_p = self.dataset.id2desc[torch.tensor(p_id)]       # batch_size * negative_size
        desc_c = self.dataset.id2desc[torch.tensor(c_id)]
        desc_s = self.dataset.id2desc[torch.tensor(s_id)]
        desc_q = self.dataset.id2desc[query_node_id]

        sentences_query = []
        sentences_pos = []
        for idx in range(query_node_id.shape[0]):       # for each sample
            # query
            sentences_query.append(f"The meaning of \"{desc_q[idx]}\" is [MASK]")
            # position
            desc = desc_p[idx] + '\n' + desc_c[idx] + '\n' + desc_s[idx]
            sentences_pos.append(f"The meaning of \"{desc}\" is [MASK]")

        encoded_query = self.tokenizer(sentences_query, padding=True, truncation=True, return_tensors='pt')
        encoded_pos = self.tokenizer(sentences_pos, padding=True, truncation=True, return_tensors='pt')
    
        return encoded_query, encoded_pos, target

    def __str__(self):
        return "\n\t".join([
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
        ])
    

def get_dataloader_unsup(data_path, taxonomy_name, shuffle=True, sampling_mode=0, batch_size=16, negative_size=32, num_workers=8,
                       tokenizer='/codes/share/huggingface_models/bert_base_uncased'):
    raw_graph_dataset = MAGDataset(name=taxonomy_name, path=data_path,
                                raw=False, existing_partition=True)
    dataset = Dataset_Unsup(graph_dataset=raw_graph_dataset, mode="train", sampling_mode=sampling_mode, negative_size=negative_size,
                        tokenizer=tokenizer)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    