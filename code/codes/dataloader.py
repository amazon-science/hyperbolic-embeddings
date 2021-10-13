#!/usr/bin/python3
import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    A class to load a Train Dataset for translation queries.

    Attributes
    ----------
    triples : list of (head,relation,tail) triplet
    nentity : Number of Entity nodes
    nrelation : Number of Relation nodes
    negative_sample_size : Number of negative samples per positive sample
    train_ans : Answers to the training queries in the triples
    mode : here 'tail_batch', str 'tail_batch' denotes training, else testing      
    """
    def __init__(self, triples, nentity, nrelation, negative_sample_size, train_ans, mode):
        assert mode == 'tail-batch'
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples, train_ans)
        self.true_tail = train_ans
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx][0]
        head, relations, tail = positive_sample
        tail = np.random.choice(list(self.true_tail[((head, relations),)]))
        subsampling_weight = self.count[(head, relations)] 
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.true_tail[((head, relations),)], 
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([head] + [i for i in relations] + [tail])
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        '''Stacks positive_samples, negative_samples and their weights into one tensor.'''
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, true_tail, start=4):
        '''Returns frequencies of (head,relation) occurrence'''
        count = {}
        for triple, qtype in triples:
            head, relations, tail = triple
            assert (head, relations) not in count
            count[(head, relations)] = start + len(true_tail[((head, relations),)])
        return count
    
class TrainInterDataset(Dataset):
    """
    A class to load a Train Dataset for intersection queries.

    Attributes
    ----------
    triples : list of (head,relation,tail) triplet
    nentity : Number of Entity nodes
    nrelation : Number of Relation nodes
    negative_sample_size : Number of negative samples per positive sample
    train_ans : Answers to the training queries in the triples
    mode : here 'tail_batch', str 'tail_batch' denotes training, else testing      
    """
    def __init__(self, triples, nentity, nrelation, negative_sample_size, train_ans, mode):
        assert mode == 'tail-batch'
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples, train_ans)
        self.true_tail = train_ans
        self.qtype = self.triples[0][-1]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.true_tail[query], 
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        flat_query = np.array([[qi[0], qi[1][0]] for qi in query]).flatten()
        tail = np.random.choice(list(self.true_tail[query]))
        positive_sample = torch.LongTensor(list(flat_query)+[tail])
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        '''Stacks positive_samples, negative_samples and their weights into one tensor.'''
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, true_tail, start=4):
        '''Returns frequencies of intersection query or (head1,head2) occurrence'''
        count = {}
        for triple in triples:
            query = triple[:-2]
            assert query not in count
            count[query] = start + len(true_tail[query])
        return count

class TestInterDataset(Dataset):
    """
    A class to load a Test Dataset for intersection queries.

    Attributes
    ----------
    triples : list of (head,relation,tail) triplet
    nentity : Number of Entity nodes
    nrelation : Number of Relation nodes
    negative_sample_size : Number of negative samples per positive sample
    train_ans : Answers to the training queries in the triples
    mode : here None, str 'tail_batch' denotes training, else testing   
    """
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        tail = self.triples[idx][-2]
        negative_sample = torch.LongTensor(range(self.nentity))
        flat_query = np.array([[qi[0], qi[1][0]] for qi in query]).flatten()
        positive_sample = torch.LongTensor(list(flat_query)+[self.triples[idx][-2]])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        '''Stacks positive_samples, negative_samples into one tensor.'''
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query

class TestChainInterDataset(Dataset):
    """
    A class to load a Test Dataset for compound queries.

    Attributes
    ----------
    triples : list of (head,relation,tail) triplet
    nentity : Number of Entity nodes
    nrelation : Number of Relation nodes
    negative_sample_size : Number of negative samples per positive sample
    train_ans : Answers to the training queries in the triples
    mode : here None, str 'tail_batch' denotes training, else testing   
    """
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        tail = self.triples[idx][-2]
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor([query[0][0], query[0][1][0], query[0][1][1], query[1][0], query[1][1][0]]+[self.triples[idx][-2]])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        '''Stacks positive_samples, negative_samples into one tensor.'''
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query

class TestInterChainDataset(Dataset):
    """
    A class to load a Test Dataset for compound queries.

    ...

    Attributes
    ----------
    triples : list of (head,relation,tail) triplet
    nentity : Number of Entity nodes
    nrelation : Number of Relation nodes
    negative_sample_size : Number of negative samples per positive sample
    train_ans : Answers to the training queries in the triples
    mode : here None, str 'tail_batch' denotes training, else testing   
    """
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        tail = self.triples[idx][-2]
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor([query[0][0], query[0][1][0], query[1][0], query[1][1][0], query[2]]+[self.triples[idx][-2]])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        '''Stacks positive_samples, negative_samples into one tensor.'''
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query

class TestDataset(Dataset):
    """
    A class to load a Test Dataset for translation queries.

    Attributes
    ----------
    triples : list of (head,relation,tail) triplet
    nentity : Number of Entity nodes
    nrelation : Number of Relation nodes
    negative_sample_size : Number of negative samples per positive sample
    train_ans : Answers to the training queries in the triples
    mode : here None, str 'tail_batch' denotes training, else testing   
    """
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relations, tail = self.triples[idx][0]
        query = ((head, relations),)
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor([head] + [rel for rel in relations] + [tail])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        '''Stacks positive_samples, negative_samples into one tensor.'''
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query
    
class BidirectionalOneShotIterator(object):
    """
    A class for bidirectional iteration over the loaded query datasets.
    Traverses from both direction of the dataset to avoid vanishing gradient. 
    Attributes
    ----------
    dataloader_head : pointer to the start of the dataset.
    dataloader_tail : pointer to the end of the dataset.
    """
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        '''Even and Odd steps traverse from top and bottom of the dataset, respectively'''
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

class SingledirectionalOneShotIterator(object):
    """
    An iterator for singledirection iteration over the loaded query datasets.
    
    Attributes
    ----------
    dataloader_head : pointer to the bottom of the dataset.
    """
    def __init__(self, dataloader_tail, qtype):
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.qtype = qtype
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
