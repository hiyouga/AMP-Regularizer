import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class Vocab:

    def __init__(self, vocab_list, add_pad=True, add_unk=True):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad: # pad_id should be zero (for mask)
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._vocab_dict[self.pad_word] = self.pad_id
            self._length += 1
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._vocab_dict[self.unk_word] = self.unk_id
            self._length += 1
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, idx):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(idx, self.unk_word)
        return self._reverse_vocab_dict[idx]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length


class Tokenizer:

    def __init__(self, vocab, dataset):
        maxlen_dict = {
            'imdb': 1065,
            'yelp13': 560,
            'yelp14': 590
        }
        self.vocab = vocab
        self.maxlen = maxlen_dict[dataset]

    @classmethod
    def from_corpus(cls, corpus, dataset):
        all_tokens = set()
        for tokens in corpus:
            all_tokens.update(tokens)
        return cls(vocab=Vocab(all_tokens), dataset=dataset)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def to_sequence(self, tokens, reverse=False, padding='post', truncating='post'):
        sequence = [self.vocab.word_to_id(t) for t in tokens]
        if reverse:
            sequence.reverse()
        return self.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.maxlen, padding=padding, truncating=truncating)


class NLPDataset(Dataset):

    def __init__(self, samples, tokenizer, dataset, split):
        data_file = os.path.join('dats', f"{dataset}_{split}.dat")
        if os.path.exists(data_file):
            print(f"loading dataset: {data_file}")
            dataset = pickle.load(open(data_file, 'rb'))
        else:
            print('building dataset...')
            dataset = [(tokenizer.to_sequence(tokens), int(label)) for tokens, label in samples]
            pickle.dump(dataset, open(data_file, 'wb'))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def load_nlp_data(batch_size, workers, dataset, data_target_dir):
    train_samples = parse_nlp_data(data_target_dir, 'train.txt')
    test_samples = parse_nlp_data(data_target_dir, 'test.txt')
    tokenizer = build_tokenizer(samples=(train_samples+test_samples), dataset=dataset)
    train_set = NLPDataset(samples=train_samples, tokenizer=tokenizer, dataset=dataset, split='train')
    test_set = NLPDataset(samples=test_samples, tokenizer=tokenizer, dataset=dataset, split='test')
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_dataloader, test_dataloader, tokenizer


def load_cv_data(data_aug, batch_size, workers, dataset, data_target_dir):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'tiny-imagenet-200':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    else:
        assert False, f"Unknow dataset : {dataset}"

    if data_aug:
        if dataset == 'svhn':
            train_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
        else:
            train_transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
    else:
        train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
        test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_dataloader, test_dataloader


def parse_nlp_data(data_target_dir, fname):
    with open(os.path.join(data_target_dir, fname), 'r', encoding='utf-8') as fin:
        samples = list()
        for line in fin.read().strip().split('\n'):
            pos = line.find(':')
            label = int(line[0:pos])
            words = line[pos+1:].strip().split()
            tokens = [w.lower() for w in words]
            samples.append((tokens, label))
        return samples


def build_tokenizer(samples, dataset):
    data_file = os.path.join('dats', f"{dataset}_tokenizer.dat")
    if os.path.exists(data_file):
        print(f"loading tokenizer: {data_file}")
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        print('building tokenizer...')
        tokenizer = Tokenizer.from_corpus(corpus=[tokens for tokens, _ in samples], dataset=dataset)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer
