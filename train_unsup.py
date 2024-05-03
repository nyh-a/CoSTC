import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sentence_transformers import losses, util, models
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
from datetime import datetime
import pickle as pkl
import math

from data_loader.data_loaders import get_dataloader
from data_loader.dataset import Dataset_random
import model.metric as module_metric
from utils.util import calc_metrics

name = "food"
if name == "food":
    data_path = "data/SemEval-Food/semeval_food.pickle.bin"
    taxonomy_name = "semeval_food"
elif name == "verb":
    data_path = "data/SemEval-Verb/wordnet_verb.pickle.bin"
    taxonomy_name = "wordnet_verb"
elif name == "mesh":
    data_path = "data/mesh/mesh.pickle.bin"
    taxonomy_name = "mesh"
else:
    raise FileNotFoundError("No such dataset " + name)


data_loader = get_dataloader(data_path, taxonomy_name, training=True, sampling_mode=0, batch_size=16,
                            negative_size=32, num_workers=1)
dataset = data_loader.dataset


def train(model_save_path):
    # prepare model
    model_name = "/codes/share/huggingface_models/bert_base_uncased"
    # model_name = "/codes/share/huggingface_models/all-mpnet-base-v2"
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if torch.cuda.is_available():
        model = SentenceTransformer(model_name, device='cuda')
    else:
        model = SentenceTransformer(model_name)
    num_epochs = 1
    train_batch_size = 80


    ######### Read train data  ##########
    # Read train data
    print("Read train data")

    # all query description
    queries = dataset.node_list
    queries_desc = [query.description for query in queries]

    # all position description
    candidate_positions = dataset.all_edges
    positions = []
    for p, c in candidate_positions:
        s = dataset._get_sibling(p, c)
        positions.append(p.description + '\n' + c.description + '\n' + s.description)
    
    corpus = queries_desc + positions
    train_samples = [InputExample(texts=[s, s]) for s in queries_desc]

    print("corpus len:",len(train_samples))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)


    # Configure the training.
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up

    # Train the model
    print("Train the model")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': 5e-5},
            checkpoint_path=model_save_path,
            output_path=model_save_path
            )
    
    # seperate
    # query_samples = [InputExample(texts=[s, s]) for s in queries_desc]
    # query_dataloader = DataLoader(query_samples, shuffle=True, batch_size=train_batch_size)
    # pos_samples = [InputExample(texts=[s, s]) for s in positions]
    # pos_dataloader = DataLoader(pos_samples, shuffle=True, batch_size=train_batch_size)
    # model.fit(train_objectives=[(query_dataloader, train_loss), (pos_dataloader, train_loss)],
    #     epochs=num_epochs,
    #     warmup_steps=warmup_steps,
    #     optimizer_params={'lr': 5e-5},
    #     checkpoint_path=model_save_path,
    #     output_path=model_save_path
    #     )


# Test

def test(saved_path):
    # load model and data
    print("Test")
    model = SentenceTransformer(saved_path, device='cuda')
    queries = dataset.test_node_list
    node2pos = dataset.test_node2pos
    candidate_positions = dataset.all_edges
    id_positions = []

    # calculate embedding
    corpus = []
    for p, c in candidate_positions:
        s = dataset._get_sibling(p, c)
        corpus.append(p.description + c.description + s.description)
        id_positions.append((dataset.taxon2id[p], dataset.taxon2id[c]))
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=False)

    queries_desc = [query.description for query in queries]
    emb_q = model.encode(queries_desc, convert_to_tensor=True, show_progress_bar=False)

    hits = util.semantic_search(emb_q, corpus_embeddings, score_function=util.dot_score, top_k=len(candidate_positions))
    

    # find leaf and nonleaf
    leaf_queries = []
    for i, query in enumerate(queries):
        poses = node2pos[query]
        flag = True
        for pos in poses:
            if pos[1] != dataset.pseudo_leaf_node:
                flag = False
                break
        if flag:
            leaf_queries.append(query)

    
    all_ranks = []
    leaf_ranks, nonleaf_ranks = [], []
    pred_rank = {}
    for i, hit in enumerate(hits):  # for each query
        query = queries[i]
        edge_rank = [s['corpus_id'] for s in hit]

        true_pos_id = [dataset.all_edges.index(pos) for pos in node2pos[query]]
        ranks = [edge_rank.index(pos)+1 for pos in true_pos_id]
        all_ranks.append(ranks)
        if query in leaf_queries:
            leaf_ranks.append(ranks)
        else:
            nonleaf_ranks.append(ranks)

    print("Calculate metrics")
    calc_metrics(all_ranks, leaf_ranks, nonleaf_ranks)




if __name__ == '__main__':
    print(taxonomy_name)
    # train
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    model_save_path = "saved/" + taxonomy_name + "/training_Unsup/" + run_id + "/bert_base_uncased"
    # model_save_path = "saved/" + taxonomy_name + "/training_Unsup/" + run_id + "/all-mpnet-base-v2"
    train(model_save_path)
    print(taxonomy_name)
    test(model_save_path)
    
    # test
    # model = "/codes/a/XMC_taxo/saved/semeval_food/training_OnlineConstrativeLoss/1030_215641"
    # test(model)