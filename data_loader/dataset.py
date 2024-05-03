import os
import pickle 
import random
import time
from collections import deque
from itertools import chain, product
import numpy as np
from abc import abstractmethod
import scipy.sparse as smat

import networkx as nx
# import spacy
from networkx.algorithms import descendants
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import neighborhood


MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000

# nlp = spacy.load('en_core_web_md')

#LIU
import torch
import numpy as np
# from torch_geometric.utils.undirected import to_undirected
# from torch_geometric.utils import remove_self_loops
# def get_edges(edge_index_list):
#     undirected_edge_list = []
#     edge_index, _ = remove_self_loops(
#         torch.from_numpy(np.array(edge_index_list)).transpose(1, 0))  # remove self-loop
#     undirected_edge_list.append(to_undirected(edge_index))  # convert to undirected/bi-directed edge_index
#     return undirected_edge_list[0]

def get_edges(edge_index_list):
    undirected_edge_list = []
    edge_index_list = [[item[0], item[1]] for item in edge_index_list if item[0] != item[1]]
    for item in edge_index_list:
        if [item[1], item[0]] not in edge_index_list:
            edge_index_list.append([item[1], item[0]])
    edge_index = torch.from_numpy(np.array(edge_index_list)).transpose(1, 0)
    # edge_index, _ = remove_self_loops(
    #     torch.from_numpy(np.array(edge_index_list)).transpose(1, 0))  # remove self-loop
    undirected_edge_list.append(edge_index) # convert to undirected/bi-directed edge_index
    return undirected_edge_list[0]


class Taxon(object):
    def __init__(self, tx_id, rank=-1, norm_name="none", display_name="None", main_type="", level="-100", p_count=0,
                 c_count=0, create_date="None", use_wordnet=True, single_word=False):
        self.tx_id = tx_id
        self.rank = int(rank)
        self.norm_name = norm_name
        self.display_name = display_name
        self.main_type = main_type
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)
        self.create_date = create_date
        self.use_wordnet = use_wordnet
        self.single_word = single_word
        self.description = ''
        # self.nlp_description = nlp('')

    def set_description(self, description):
        self.description = description
        # self.nlp_description = nlp(description)

    def __str__(self):
        return "Taxon {} (name: {}, level: {})".format(self.tx_id, self.norm_name, self.level)

    def __lt__(self, other):
        if self.display_name < other.display_name:
            return True
        else:
            return False

    def __eq__(self, other):
        if isinstance(other, Taxon):
            return self.display_name == other.display_name
        return False


    def __hash__(self):
        return hash(self.display_name)


class MAGDataset(object):
    def __init__(self, name, path, embed_suffix="", raw=True, existing_partition=True, partition_pattern='internal'):
        """ Raw dataset class for MAG dataset

        Parameters
        ----------
        name : str
            taxonomy name
        path : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        embed_suffix : str
            suffix of embedding file name, by default ""
        raw : bool, optional
            load raw dataset from txt (True) files or load pickled dataset (False), by default True
        existing_partition : bool, optional_my
            whether to use the existing the train/validation/test partitions or randomly sample new ones, by default False
        """
        self.name = name  # taxonomy name
        self.embed_suffix = embed_suffix
        self.existing_partition = existing_partition
        self.partition_pattern = partition_pattern
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids
        self.test_node_ids = []  # a list of test node_ids
        self.data_path = path

        if raw:
            self._load_dataset_raw(path)
        else:
            self._load_dataset_pickled(path)

    def _load_dataset_pickled(self, pickle_path):
        print('loading pickled dataset')
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
        self.name = data["name"]
        self.taxonomy = data['taxonomy']
        self.node_id2taxon = data['id2taxon']
        self.taxon2node_id = data['taxon2id']
        self.vocab = data["vocab"]
        self.train_node_ids = data["train_node_ids"]
        self.validation_node_ids = data["validation_node_ids"]
        self.test_node_ids = data["test_node_ids"]

        # path = os.path.split(pickle_path)[0]
        #
        # with open(os.path.join(path, f'{self.name}.terms.train'), 'w') as f:
        #     for id in self.train_node_ids:
        #         f.write(str(id) + '\n')
        # with open(os.path.join(path, f'{self.name}.terms.validation'), 'w') as f:
        #     for id in self.validation_node_ids:
        #         f.write(str(id) + '\n')
        # with open(os.path.join(path, f'{self.name}.terms.test'), 'w') as f:
        #     for id in self.test_node_ids:
        #         f.write(str(id) + '\n')
        # exit(0)
        print('dataset loaded')

    def _load_dataset_raw(self, dir_path):
        path = os.path.split(dir_path)[0]
        node_file_name = os.path.join(path, f"{self.name}.terms")
        edge_file_name = os.path.join(path, f"{self.name}.taxo")
        desc_file_name = os.path.join(path, f"{self.name}.desc")
        if self.embed_suffix == "":
            output_pickle_file_name = os.path.join(path, f"{self.name}.pickle.bin")
        else:
            output_pickle_file_name = os.path.join(path, f"{self.name}.{self.embed_suffix}.pickle.bin")

        tx_id2taxon = {}
        self.taxonomy = nx.DiGraph()

        # load nodes
        with open(desc_file_name, "r", encoding='utf-8') as fdesc:
            with open(node_file_name, "r", encoding='utf-8') as fin:
                for line, desc in tqdm(zip(fin, fdesc), desc="Loading terms"):
                    line = line.strip()
                    desc = desc.strip()
                    if line:
                        segs = line.split("\t")
                        segs_desc = desc.split("\t")
                        assert len(segs) == 2, f"Wrong number of segmentations {line}"
                        try:
                            assert segs[1] == segs_desc[0]
                            desc = segs_desc[1]
                        except AssertionError:
                            # assert len(segs_desc) == 1
                            desc = segs_desc[0]
                        taxon = Taxon(tx_id=segs[0], norm_name=segs[1], display_name=segs[1])
                        taxon.set_description(desc)
                        tx_id2taxon[segs[0]] = taxon
                        self.taxonomy.add_node(taxon)
        # load edges
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2taxon[segs[0]]
                    child_taxon = tx_id2taxon[segs[1]]
                    self.taxonomy.add_edge(parent_taxon, child_taxon)

        # generate vocab
        # tx_id is the old taxon_id read from {self.name}.terms file, node_id is the new taxon_id from 0 to len(vocab)
        self.tx_id2node_id = {node.tx_id: idx for idx, node in enumerate(self.taxonomy.nodes())}
        self.node_id2tx_id = {v: k for k, v in self.tx_id2node_id.items()}
        self.node_id2taxon = {k: tx_id2taxon[v] for k, v in self.node_id2tx_id.items()}
        self.taxon2node_id = {v: k for k, v in self.node_id2taxon.items()}
        self.vocab = [tx_id2taxon[self.node_id2tx_id[node_id]].norm_name + "@@@" + str(node_id) for node_id in
                      self.node_id2tx_id]

        if self.existing_partition:
            # Use the pickled partitions
            # with open(os.path.join(path, 'split_node_ids.pickle'), 'rb') as f:
            #     data = pickle.load(f)
            # self.validation_node_ids = data["validation_node_ids"]
            # self.test_node_ids = data["test_node_ids"]
            # self.train_node_ids = data["train_node_ids"]

            # Or use the train/val/test files
            dir_path = os.path.dirname(dir_path)
            train_node_file_name = os.path.join(dir_path, f"{self.name}.terms.train")
            validation_node_file_name = os.path.join(dir_path, f"{self.name}.terms.validation")
            test_file_name = os.path.join(dir_path, f"{self.name}.terms.test")

            raw_train_node_list = self._load_node_list(train_node_file_name)
            raw_validation_node_list = self._load_node_list(validation_node_file_name)
            raw_test_node_list = self._load_node_list(test_file_name)

            self.train_node_ids = [int(n) for n in raw_train_node_list]
            self.validation_node_ids = [int(n) for n in raw_validation_node_list]
            self.test_node_ids = [int(n) for n in raw_test_node_list]

        else:
            print("Partition graph ...")
            if self.partition_pattern == 'leaf':
                sampled_node_ids = []
                for node in self.taxonomy.nodes():
                    if self.taxonomy.out_degree(node) == 0:
                        sampled_node_ids.append(self.tx_id2node_id[node.tx_id])
                random.seed(47)
                random.shuffle(sampled_node_ids)
            elif self.partition_pattern == 'internal':
                root_node = [node for node in self.taxonomy.nodes() if self.taxonomy.in_degree(node) == 0]
                sampled_node_ids = [self.tx_id2node_id[node.tx_id] for node in self.taxonomy.nodes() if
                                    node not in root_node]
                random.seed(47)
                random.shuffle(sampled_node_ids)
            else:
                raise ValueError('Unknown partition method!')

            validation_size = min(int(len(sampled_node_ids) * 0.1), MAX_VALIDATION_SIZE)
            test_size = min(int(len(sampled_node_ids) * 0.1), MAX_TEST_SIZE)
            self.validation_node_ids = sampled_node_ids[:validation_size]
            self.test_node_ids = sampled_node_ids[validation_size:(validation_size + test_size)]
            self.train_node_ids = [node_id for node_id in self.node_id2tx_id if
                                   node_id not in self.validation_node_ids and node_id not in self.test_node_ids]
            print("Finish partitioning graph ...")

        # save to pickle for faster loading next time
        print("start saving pickle data")
        with open(output_pickle_file_name, 'wb') as fout:
            data = {
                "name": self.name,
                "taxonomy": self.taxonomy,
                "id2taxon": self.node_id2taxon,
                "taxon2id": self.taxon2node_id,
                "vocab": self.vocab,
                "train_node_ids": self.train_node_ids,
                "validation_node_ids": self.validation_node_ids,
                "test_node_ids": self.test_node_ids,
            }
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def _load_node_list(self, file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line)
        return node_list


class RawDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32):
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size

        self.taxon2id = graph_dataset.taxon2node_id
        self.id2taxon = graph_dataset.node_id2taxon
        train_nodes = [self.id2taxon[node_id] for node_id in graph_dataset.train_node_ids]

        # add pseudo root
        full_graph = graph_dataset.taxonomy     
        roots = [node for node in full_graph.nodes() if full_graph.in_degree(node) == 0]
        self.pseudo_root_node = Taxon(tx_id='', norm_name='pseudo root', display_name='pseudo root')
        full_graph.add_node(self.pseudo_root_node)
        for node in roots:
            full_graph.add_edge(self.pseudo_root_node, node)
        train_nodes.append(self.pseudo_root_node)
        self.full_graph = full_graph

        if mode == 'train':
            # add pseudo leaf node to core graph
            datapath = os.path.split(graph_dataset.data_path)[0]
            graph_pickle_path = os.path.join(datapath, 'subgraphs.pickle')
            graph_pickled = False
            if os.path.isfile(graph_pickle_path):
                graph_pickled = True
                with open(graph_pickle_path, 'rb') as f:
                    graphs = pickle.load(f)

            print('adding pseudo leaf')
            if graph_pickled:
                self.core_subgraph = graphs['core_subgraph']
                self.pseudo_leaf_node = graphs['pseudo_leaf_node']
            else:
                self.core_subgraph = self._get_holdout_subgraph(train_nodes)
                self.pseudo_leaf_node = Taxon(tx_id='', norm_name='pseudo leaf', display_name='pseudo leaf')
                self.core_subgraph.add_node(self.pseudo_leaf_node)
                for node in list(self.core_subgraph.nodes()):
                    self.core_subgraph.add_edge(node, self.pseudo_leaf_node)

            
            self.taxon2id[self.pseudo_leaf_node] = len(full_graph.nodes)
            self.taxon2id[self.pseudo_root_node] = len(full_graph.nodes) - 1
            self.id2taxon[len(full_graph.nodes)] = self.pseudo_leaf_node
            self.id2taxon[len(full_graph.nodes) - 1] = self.pseudo_root_node
            self.leaf_nodes = [node for node in self.core_subgraph.nodes() if self.core_subgraph.out_degree(node) == 1]

            self.id2desc = np.array([taxon.description for id, taxon in self.id2taxon.items()])

            # add interested node list and subgraph
            # remove supersource nodes (i.e., nodes without in-degree 0)
            self.node_list = [n for n in train_nodes if n != self.pseudo_root_node]

            # build node2pos, node2edge
            print('building node2pos, node2edge')
            self.node2pos, self.node2edge = {}, {}
            self.node2parents, self.node2children = {}, {}
            for node in self.node_list:
                parents = set(self.core_subgraph.predecessors(node))
                children = set(self.core_subgraph.successors(node))
                if len(children) > 1:
                    children = [i for i in children if i != self.pseudo_leaf_node]
                node_pos_edges = [(pre, suc) for pre in parents for suc in children if pre != suc]
                if len(node_pos_edges) == 0:
                    node_pos_edges = [(pre, suc) for pre in parents for suc in children]

                self.node2edge[node] = set(self.core_subgraph.in_edges(node)).union(
                    set(self.core_subgraph.out_edges(node)))
                self.node2pos[node] = node_pos_edges
                self.node2parents[node] = parents
                self.node2children[node] = children


            print('building valid and test node list')
            self.valid_node_list = [self.id2taxon[node_id] for node_id in graph_dataset.validation_node_ids]
            if graph_pickled:
                self.valid_holdout_subgraph = graphs['valid_subgraph']
            else:
                self.valid_holdout_subgraph = self._get_holdout_subgraph(train_nodes + self.valid_node_list)
                self.valid_holdout_subgraph.add_node(self.pseudo_leaf_node)
                for node in [node for node in self.valid_holdout_subgraph.nodes() if
                             self.valid_holdout_subgraph.out_degree(node) == 0]:
                    self.valid_holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.valid_id2taxon = {idx: taxon for idx, taxon in enumerate(self.valid_holdout_subgraph.nodes())}
            self.valid_taxon2id = {v: k for k, v in self.valid_id2taxon.items()}
            self.valid_node2pos = self._find_insert_position(self.valid_node_list, self.valid_holdout_subgraph)

            self.test_node_list = [self.id2taxon[node_id] for node_id in graph_dataset.test_node_ids]
            if graph_pickled:
                self.test_holdout_subgraph = graphs['test_subgraph']
            else:
                self.test_holdout_subgraph = self._get_holdout_subgraph(train_nodes + self.test_node_list)
                self.test_holdout_subgraph.add_node(self.pseudo_leaf_node)
                for node in [node for node in self.test_holdout_subgraph.nodes() if
                             self.test_holdout_subgraph.out_degree(node) == 0]:
                    self.test_holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.test_id2taxon = {idx: taxon for idx, taxon in enumerate(self.test_holdout_subgraph.nodes())}
            self.test_taxon2id = {v: k for k, v in self.test_id2taxon.items()}
            self.test_node2pos = self._find_insert_position(self.test_node_list, self.test_holdout_subgraph)

            if not graph_pickled:
                with open(graph_pickle_path, 'wb') as f:
                    pickle.dump({
                        'pseudo_leaf_node': self.pseudo_leaf_node,
                        'core_subgraph': self.core_subgraph,
                        'valid_subgraph': self.valid_holdout_subgraph,
                        'test_subgraph': self.test_holdout_subgraph
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # used for sampling negative positions during train/validation stage
            self.pointer = 0
            self.all_edges = list(self._get_all_candidate_positions(self.core_subgraph))
            random.shuffle(self.all_edges)
            self.all_edge_id = [(self.taxon2id[edge[0]], self.taxon2id[edge[1]]) for edge in self.all_edges]
            self.all_edge_id.sort()
            
            # TODO: for fixed sibling training
            self.pos_id2desc = []

            # self.all_pos = list(self._get_candidate_positions(self.core_subgraph, self.core_subgraph.nodes))
            self.all_pos = None
            

            self.node2pos_node = {}
            tot = 0
            for node, eles in self.node2pos.items():
                self.node2pos_node[node] = [set(), set()]
                # xu:正例只有一边时只加入前面
                if len(eles) == 1 and eles[0][1] is self.pseudo_leaf_node:
                    self.node2pos_node[node][0].add(eles[0][0])
                    tot += 1
                    continue
                for ele in eles:
                    self.node2pos_node[node][0].add(ele[0])
                    self.node2pos_node[node][1].add(ele[1])
            print(tot, len(self.node2pos))
            for node, eles in self.node2edge.items():
                for ele in eles:
                    self.node2pos_node[node][0].add(ele[0])
                    self.node2pos_node[node][1].add(ele[1])
            
        end = time.time()
        print(f"Finish loading dataset ({end - start} seconds)")

    def __str__(self):
        return f"{self.__class__.__name__} mode:{self.mode}"

    def __len__(self):
        return len(self.node_list)

    @abstractmethod
    def __getitem__(self, idx):
        """
        Generate an data instance based on train/validation/test mode.
            
        """
        raise NotImplementedError


    def load_indexer(self, indexer_path="/codes/a/XMC_taxo/data/SemEval-Food/indexer/sbert/code.npz"):
        self.indexer = smat.load_npz(indexer_path)
        # indexer.shape = n_positions * num_cluster
        self.n_pos = self.indexer.shape[0]
        self.n_groups = self.indexer.shape[1]

        # position2cluster = indexer[label].argmax()
        self.map_group_y = np.zeros(self.n_pos, dtype=int)
        for label in range(self.indexer.shape[0]):
            self.map_group_y[label] = self.indexer[label].argmax()
        
        # cluster2postions
        self.group_y = []   # num_cluster*[num_positions]   average lables per cluster=
        for i in range(self.indexer.shape[1]):
            labels = self.indexer[:,i].nonzero()[0]
            self.group_y.append(labels)


    def _get_holdout_subgraph(self, nodes):
        node_to_remove = [n for n in self.full_graph.nodes if n not in nodes]
        subgraph = self.full_graph.subgraph([node for node in nodes]).copy()
        for node in node_to_remove:
            parents = set()
            children = set()
            ps = deque(self.full_graph.predecessors(node))
            cs = deque(self.full_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(self.full_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(self.full_graph.successors(c))
            for p, c in product(parents, children):
                subgraph.add_edge(p, c)
        # remove jump edges
        node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                successors1 = set(subgraph.successors(node))
                successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
                checkset = successors1.intersection(successors2)
                if checkset:
                    for s in checkset:
                        # if subgraph.in_degree(s) > 1:
                        subgraph.remove_edge(node, s)
        return subgraph

    def _get_all_candidate_positions(self, graph):
        node2descendants = {n: set(descendants(graph, n)) for n in graph.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates
    
    def _get_candidate_positions(self, graph, care_nodes):
        # one-hop and two-hop candidates  return taxon (p, c)
        # node2descendants = {n: set(graph.neighbors(n)) for n in care_nodes}
        node2descendants = {}
        for n in care_nodes:
            neighbors = neighborhood(graph, n, 1) + neighborhood(graph, n, 2)
            node2descendants[n] = set(neighbors) 
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates
    
    def _get_candidate_positions_with_sibling(self, graph, care_nodes):
        # one-hop and two-hop candidates   return node_id (p,c,s)
        node2descendants = {}
        for n in care_nodes:
            neighbors = neighborhood(graph, n, 1) + neighborhood(graph, n, 2)
            node2descendants[n] = set(neighbors) 
        candidates = list(set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()])))
        candidates_id = [(self.taxon2id[p], self.taxon2id[c], self.taxon2id[self._get_sibling(p,c)]) for p,c in candidates]
        return candidates_id
    
    def _get_sibling(self, p, c):
        # return a taxon
        if p == self.pseudo_root_node:   # reasonable ?
            if c == self.pseudo_leaf_node:
                return self.pseudo_leaf_node
            if self.fixed:
                sibling = list(self.node2parents[c])[0]
            sibling = random.choice(list(self.node2parents[c]))
        else:
            if self.fixed:
                sibling = list(self.node2children[p])[0]
            sibling = random.choice(list(self.node2children[p])) if len(self.node2children[p])>1 else self.pseudo_leaf_node
        return sibling

    def _find_insert_position(self, node_ids, holdout_graph, ignore=[]):
        node2pos = {}
        subgraph = self.core_subgraph
        for node in node_ids:
            if node in ignore:
                continue
            parents = set()
            children = set()
            ps = deque(holdout_graph.predecessors(node))
            cs = deque(holdout_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(holdout_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(holdout_graph.successors(c))
            if not children:
                children.add(self.pseudo_leaf_node)
            position = [(p, c) for p in parents for c in children if p != c]
            node2pos[node] = position
        return node2pos




        
class Dataset_random2(RawDataset):
    """
    Random negative sampling    all pos + N neg     return tuple
    """
    def __init__(self, graph_dataset, mode="train", sampling_mode=0, negative_size=4):
        super().__init__(graph_dataset, mode, sampling_mode, negative_size)
        self.fixed = False


    def __getitem__(self, idx):
        # positive
        query_node = self.node_list[idx]
        query_node_id = self.taxon2id[query_node]
        positions = list(self.node2pos[query_node])

        # negative
        if self.negative_size < len(positions):
            positions = random.sample(positions, self.negative_size//2)
        negative_size = self.negative_size - len(positions)

        positions += self._get_k_negatives(query_node, negative_size)

        res = []
        for p, c in positions:
            s = self._get_sibling(p, c)
            res.append( [query_node_id, self.taxon2id[p], self.taxon2id[c], self.taxon2id[s],
                       int(p in self.node2parents[query_node]),
                    int(c in self.node2children[query_node]),
                    int( (p in self.node2parents[query_node]) & (c in self.node2children[query_node]) )
            ] )
            
        return tuple(res)

        

    def _get_k_negatives(self, query_node, negative_size, ignore=[]):
        """ 
        Generate EXACTLY negative_size samples for the query node
        """
        positions = None
        # from self.all_edge
        if self.sampling_mode == 0:
            positions = self.all_edges
        # from self.all_pos
        elif self.sampling_mode == 1:
            positions = self.all_pos

        if self.pointer == 0:
            random.shuffle(positions)

        negatives = []
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in positions[self.pointer: self.pointer + n_lack] if
                              ele not in self.node2pos[query_node] and ele not in self.node2edge[
                                  query_node] and ele not in ignore])
            self.pointer += n_lack
            if self.pointer >= len(self.all_edges):
                self.pointer = 0
                random.shuffle(self.all_edges)
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives
    





class Dataset_Diversity(RawDataset):
    """
    For diversity negative sampling   return all pos + N neg     return tuple
    """
    def __init__(self, graph_dataset, mode="train", sampling_mode=0, negative_size=4, fixed=False):
        super().__init__(graph_dataset, mode, sampling_mode, negative_size)
        self.fixed = fixed
        print("Constructing node_dist_matrix")
        G = self.core_subgraph.to_undirected()
        G.remove_node(self.pseudo_leaf_node)
        path = os.path.split(graph_dataset.data_path)[0]
        node_dist_mat_file_name = os.path.join(path, "node_dist_mat.pickle")
        if os.path.exists(node_dist_mat_file_name):
            with open(node_dist_mat_file_name, 'rb') as f:
                self.node_dist_mat = pickle.load(f)
        else:
            self.node_dist_mat = nx.floyd_warshall_numpy(G, nodelist=G.nodes())
        self.dist_id2taxon ={idx: taxon for idx, taxon in enumerate(G.nodes())}
        self.dist_taxon2id = {taxon: idx for idx, taxon in enumerate(G.nodes())}
        print("Constructing pos and edges")
        self.all_pos = list(self._get_candidate_positions(self.core_subgraph, self.core_subgraph.nodes))

        self.jump_edges = list(set(self.all_edges)- set(self.all_pos))

        self.node2descendants = {n: set(descendants(self.core_subgraph, n)) for n in self.core_subgraph.nodes}

        self.ratio = 0.25
        # self.ratio = 1


    def __getitem__(self, idx):
        # positive
        query_node = self.node_list[idx]
        query_node_id = self.taxon2id[query_node]
        positions = list(self.node2pos[query_node])

        # negative
        if self.negative_size < len(positions):
            positions = random.sample(positions, self.negative_size//2)
        negative_size = self.negative_size - len(positions)

        positions += self._get_k_negatives(query_node, negative_size)

        res = []
        for p, c in positions:
            s = self._get_sibling(p, c)
            res.append( [query_node_id, self.taxon2id[p], self.taxon2id[c], self.taxon2id[s],
                       int(p in self.node2parents[query_node]),
                    int(c in self.node2children[query_node]),
                    int( (p in self.node2parents[query_node]) & (c in self.node2children[query_node]) )
            ] )
            
        return tuple(res)

        

    def _get_k_negatives(self, query_node, negative_size, ignore={0, 1}):
        # sample for diversity
        q_dist_id = self.dist_taxon2id[query_node]
        q_dist = self.node_dist_mat[q_dist_id]
        all_dist = set(q_dist)

        negatives = []
        # while len(negatives) < negative_size:
        #     dist = random.sample(all_dist - ignore, 1)[0]
        #     dist_node_id = np.where(q_dist == dist)[0].tolist()
        #     node1 = self.dist_id2taxon[random.sample(dist_node_id, 1)[0]]

        #     descendants = self.node2descendants[node1]
        #     node2 = random.sample(descendants, 1)[0]
        #     if node1 == self.pseudo_leaf_node or node2 == self.pseudo_root_node:
        #         continue
        #     negatives.append((node1, node2))


        
        # with topological diversity
        while len(negatives) < negative_size * self.ratio:
            dist = random.sample(all_dist - ignore, 1)[0]
            dist_node_id = np.where(q_dist == dist)[0].tolist()
            node = self.dist_id2taxon[random.sample(dist_node_id, 1)[0]]
            if node == self.pseudo_root_node:
                continue
            linked_edge = [ele for ele in self.all_pos if node==ele[0] or node==ele[1]]
            # random select edges(all_pos) linked to node 
            if len(self.node2pos[node]) > 0:
                linked_edge.append(list(self.node2pos[node])[0])
                negatives.append(random.sample(linked_edge, 1)[0])


            # all linked edge
            # linked_edge.extend(list(self.node2pos[node]))
            # negatives.extend(linked_edge)


        # without topological diversity
        # while len(negatives) < negative_size * self.ratio:
        #     negatives.append(random.sample(self.all_pos, 1)[0])

            

        # sample from jump_edge  easy
        while len(negatives) < negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in random.sample(self.jump_edges, k=n_lack) if
                                ele not in self.node2pos[query_node] and ele not in self.node2edge[
                                    query_node] and ele not in ignore])
        
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives
            



class Dataset_Unsup(RawDataset):
    """
    For query-query, position-postion contrastive training  return a text pair     return tuple
    """
    def __init__(self, graph_dataset, mode="train", sampling_mode=0, negative_size=64, tokenizer='/codes/share/huggingface_models/bert_base_uncased'):
        super().__init__(graph_dataset, mode, sampling_mode, negative_size)

        self.fixed = False
        # get all descriptions
        self._get_all_desc(sampling_mode)
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer)
        


    def __len__(self):
        if self.sampling_mode == 3:
            return (len(self.queries_desc) + len(self.pos_desc))*2 // self.negative_size
        return len(self.descs)


    def __getitem__(self, idx):
        if self.sampling_mode == 3:
            flag = random.choice([0, 1])
            if flag == 0:
                desc = random.sample(self.queries_desc, k=self.negative_size)
            else:
                desc = random.sample(self.pos_desc, k=self.negative_size)
            encoded_desc = self.tokenizer(desc, max_length=128, padding="max_length", truncation=True, return_tensors='pt')
            input_ids = encoded_desc['input_ids']
            attention_mask = encoded_desc['attention_mask']
            return input_ids, attention_mask


        desc = self.descs[idx]
        encoded_desc = self.tokenizer(desc, max_length= 128, padding="max_length", truncation=True, return_tensors='pt')
        input_ids = encoded_desc['input_ids']
        attention_mask = encoded_desc['attention_mask']

        return input_ids.squeeze(0), attention_mask.squeeze(0)

        

    def _get_k_negatives(self, query_node, negative_size, ignore={0, 1}):
        pass

    def _get_all_desc(self, sampling_mode):
        # get all descriptions
        queries_desc = [f"The meaning of \"{query.description}\" is [MASK]" for query in self.node_list]
        if sampling_mode == 0:
            # remove positions with only one description
            pos_desc = []
            for p, c in self.all_edges:
                s = self._get_sibling(p, c)
                pe = p.description != ''
                ce = c.description != ''
                se = s.description != ''
                if (pe and ce) or (pe and se) or (ce and se):
                    desc = p.description + '\n' + c.description + '\n' + s.description
                    pos_desc.append(f"The meaning of \"{desc}\" is [MASK]")
            self.descs = queries_desc + pos_desc
        elif sampling_mode == 1:
            self.descs = queries_desc
        elif sampling_mode == 2:
            pos_desc = []
            for p, c in self.all_edges:
                s = self._get_sibling(p, c)
                desc = p.description + '\n' + c.description + '\n' + s.description
                pos_desc.append(f"The meaning of \"{desc}\" is [MASK]")
            self.descs = pos_desc
        elif sampling_mode == 3:
            self.queries_desc = queries_desc
            self.pos_desc = []
            for p, c in self.all_edges:
                s = self._get_sibling(p, c)
                desc = p.description + '\n' + c.description + '\n' + s.description
                self.pos_desc.append(f"The meaning of \"{desc}\" is [MASK]")

