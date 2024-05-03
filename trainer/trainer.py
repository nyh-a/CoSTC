import numpy as np
import torch
from torch import autograd, nn
import torch.nn.functional as F
import pickle as pkl
import random
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, calc_metrics
from model.loss import *
from utils import mean_pooling
from sentence_transformers import SentenceTransformer, util


class Trainer_Prompt(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        # self.do_validation = self.valid_data_loader is not None
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.bce_loss = bceWlogit_loss
        self.scale = 20
        self.margin = self.config['trainer']['margin']
        self.curriculum = self.config['trainer']['curriculum']



    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # for curriculum learning
        if self.curriculum:
            self.data_loader.dataset.ratio += 0.005
            self.margin -= 0.005
        for batch_idx, data in enumerate(self.data_loader):
            
            # data
            encoded_query, encoded_pos, target = data
            # print(target.shape)
            target = target.to(self.device)
            # print(target)
            
            # Compute embeddings
            emb_query = self.model(**encoded_query.to(self.device))
            emb_pos = self.model(**encoded_pos.to(self.device))
            # print(emb_q.shape, emb_pos.shape)
            
            # calculate loss
            # for bceWlogits
            # scores = (util.pairwise_cos_sim(emb_query, emb_pos) - self.margin) * self.scale
            # loss = self.bce_loss(scores, target.float())
            # for contrastive loss
            loss = self.criterion(emb_query, emb_pos, target, margin=self.margin)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # log
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            
            # for debug valid_epoch
            # break
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val, leaf_val, nonleaf_val = self._valid_epoch(epoch, "valid")
            log = {**log, **val, **leaf_val, **nonleaf_val}

        # if self.lr_scheduler is not None:
        #         self.lr_scheduler.step()       
        return log

    def _valid_epoch(self, epoch, mode="valid"):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # data
            dataset = self.data_loader.dataset
            if mode == 'test':
                queries = dataset.test_node_list
                node2pos = dataset.test_node2pos
                candidate_positions = self.data_loader.dataset.all_edges
            else:
                queries = dataset.valid_node_list
                node2pos = dataset.valid_node2pos
                candidate_positions = self.data_loader.dataset.all_edges

            self.logger.info(f'number of queriess: {len(queries)}')
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            # get all descriptions
            pos_desc = []
            for p, c in candidate_positions:
                s = dataset._get_sibling(p, c)
                desc = p.description + '\n' + c.description + '\n' + s.description
                pos_desc.append(f"The meaning of \"{desc}\" is [MASK]")

            queries_desc = [f"The meaning of \"{query.description}\" is [MASK]" for query in queries]

            # calculate sentence embedding
            emb_q = self._encode(queries_desc, batch_size=128)
            print("emb_q", emb_q.shape)
            emb_pos = self._encode(pos_desc, batch_size=128)
            print("emb_pos", emb_pos.shape)

            # get all positive pairs
            pairs_x = []
            pairs_y = []
            for i in range(len(queries)):
                query = queries[i]
                true_pos_id = torch.tensor([dataset.all_edges.index(pos) for pos in node2pos[query]])
                pos_emb = emb_pos[true_pos_id.long()]
                for j in range(pos_emb.size(0)):
                    pairs_x.append(emb_q[i])
                    pairs_y.append(emb_pos[j])

            pairs_x = torch.stack(pairs_x)
            pairs_y = torch.stack(pairs_y)

            alignment = align_loss(pairs_x, pairs_y)
            uniformity = uniform_loss(torch.cat((pairs_x, pairs_y)))
            self.logger.info(f'align_loss: {alignment}')
            self.logger.info(f'uniform_loss: {uniformity}')
        

            dist_semantic = util.cos_sim(emb_q, emb_pos)  # 0-2
            print("dist_semantic", dist_semantic.shape)
        
            pred_ranks = torch.argsort(dist_semantic, dim=1, descending=True)


            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            # find leaf and nonleaf
            for i, query in enumerate(queries):
                poses = node2pos[query]
                flag = True
                for pos in poses:
                    if pos[1] != dataset.pseudo_leaf_node:
                        flag = False
                        break
                if flag:
                    leaf_queries.append(query)


            for i in range(len(queries)):  # for each query
            # for i, hit in enumerate(hits):
                query = queries[i]
                edge_rank = pred_ranks[i].tolist()
                true_pos_id = [dataset.all_edges.index(pos) for pos in node2pos[query]]
                ranks = [edge_rank.index(pos)+1 for pos in true_pos_id]

                all_ranks.append(ranks)
                if query in leaf_queries:
                    leaf_ranks.append(ranks)
                else:
                    nonleaf_ranks.append(ranks)

        return calc_metrics(all_ranks, leaf_ranks, nonleaf_ranks)
    
    def _encode(self, sentences, batch_size=64):

        all_embeddings = []
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences[start_index:start_index+batch_size]
            encoded_input = self.data_loader.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt')
            embedding = self.model(**encoded_input.to(self.device))
            all_embeddings.extend(embedding)

        return torch.stack(all_embeddings)


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    


class Trainer_Prompt_Unsup(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        # self.do_validation = self.valid_data_loader is not None
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.scale = 20.0
        self.margin = self.config['trainer']['margin']



    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # for curriculum learning
        # self.data_loader.dataset.ratio += 0.005
        for batch_idx, data in enumerate(self.data_loader):
            # data
            input_ids, attention_mask = data
            if self.config['data_loader']['args']['sampling_mode'] == 3:
                input_ids, attention_mask = input_ids.squeeze(0), attention_mask.squeeze(0)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            encoded_s = {"input_ids":input_ids, "attention_mask": attention_mask}
            

            emb_a = self.model(**encoded_s)
            emb_b = self.model(**encoded_s)

            scores = util.cos_sim(emb_a, emb_b) * self.scale
            
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
            # print(scores.shape)
            # print(labels.shape)
            # print(scores)
            # print(labels)

            loss = self.criterion(scores, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # log
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            
            # for debug valid_epoch
            # break
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val, leaf_val, nonleaf_val = self._valid_epoch(epoch, "valid")
            log = {**log, **val, **leaf_val, **nonleaf_val}

        # if self.lr_scheduler is not None:
        #         self.lr_scheduler.step()       
        return log

    def _valid_epoch(self, epoch, mode="valid"):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # data
            dataset = self.data_loader.dataset
            if mode == 'test':
                queries = dataset.test_node_list
                node2pos = dataset.test_node2pos
                candidate_positions = self.data_loader.dataset.all_edges
            else:
                queries = dataset.valid_node_list
                node2pos = dataset.valid_node2pos
                candidate_positions = self.data_loader.dataset.all_edges

            self.logger.info(f'number of queriess: {len(queries)}')
            self.logger.info(f'number of candidate positions: {len(candidate_positions)}')

            # get all descriptions
            pos_desc = []
            for p, c in candidate_positions:
                s = dataset._get_sibling(p, c)
                desc = p.description + '\n' + c.description + '\n' + s.description
                pos_desc.append(f"The meaning of \"{desc}\" is [MASK]")

            queries_desc = [f"The meaning of \"{query.description}\" is [MASK]" for query in queries]


            # calculate sentence embedding
            emb_q = self._encode(queries_desc, batch_size=128)
            print("emb_q", emb_q.shape)
            emb_pos = self._encode(pos_desc, batch_size=128)
            print("emb_pos", emb_pos.shape)

            # get all positive pairs
            pairs_x = []
            pairs_y = []
            for i in range(len(queries)):
                query = queries[i]
                true_pos_id = torch.tensor([dataset.all_edges.index(pos) for pos in node2pos[query]])
                pos_emb = emb_pos[true_pos_id.long()]
                for j in range(pos_emb.size(0)):
                    pairs_x.append(emb_q[i])
                    pairs_y.append(emb_pos[j])

            pairs_x = torch.stack(pairs_x)
            pairs_y = torch.stack(pairs_y)
            # print(pairs_x.shape)
            # print(pairs_y.shape)

            alignment = align_loss(pairs_x, pairs_y)
            uniformity = uniform_loss(torch.cat((pairs_x, pairs_y)))
            self.logger.info(f'align_loss: {alignment}')
            self.logger.info(f'uniform_loss: {uniformity}')

        
            # calculate distance
            dist_semantic = util.cos_sim(emb_q, emb_pos)  # 0-2
            print("dist_semantic", dist_semantic.shape)
        
            pred_ranks = torch.argsort(dist_semantic, dim=1, descending=True)


            all_ranks = []
            leaf_queries = []
            leaf_ranks, nonleaf_ranks = [], []

            # find leaf and nonleaf
            for i, query in enumerate(queries):
                poses = node2pos[query]
                flag = True
                for pos in poses:
                    if pos[1] != dataset.pseudo_leaf_node:
                        flag = False
                        break
                if flag:
                    leaf_queries.append(query)


            for i in range(len(queries)):  # for each query
            # for i, hit in enumerate(hits):
                query = queries[i]
                edge_rank = pred_ranks[i].tolist()
                true_pos_id = [dataset.all_edges.index(pos) for pos in node2pos[query]]
                ranks = [edge_rank.index(pos)+1 for pos in true_pos_id]

                all_ranks.append(ranks)
                if query in leaf_queries:
                    leaf_ranks.append(ranks)
                else:
                    nonleaf_ranks.append(ranks)

        return calc_metrics(all_ranks, leaf_ranks, nonleaf_ranks)
    
    def _encode(self, sentences, batch_size=64):

        all_embeddings = []
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences[start_index:start_index+batch_size]
            encoded_input = self.data_loader.dataset.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt')
            embedding = self.model(**encoded_input.to(self.device))
            all_embeddings.extend(embedding)

        return torch.stack(all_embeddings)


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)