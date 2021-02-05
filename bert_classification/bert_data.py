import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import pandas as pd
from bert_classification.bert_preprocessing import preprocess_fn, attention_mask, train_test_split_fn
from bert_classification.arg import args

train = pd.read_csv("../nsmc/ratings_train.txt", sep='\t')
test = pd.read_csv("../nsmc/ratings_test.txt", sep='\t')

train_sentences = train['document']
train_labels = train['label'].values

test_sentences = test['document']
test_labels = test['label'].values

pr_train = preprocess_fn(train_sentences,mode='train')
pr_test = preprocess_fn(test_sentences,mode='test')
test_input = pr_test # 이름 맞춰주기 위해

train_attn = attention_mask(pr_train)
test_attn = attention_mask(pr_test)
test_mask = test_attn # 이름 맞춰주기 위해

train_inputs, val_inputs, train_labels, val_labels = train_test_split_fn(pr_train,train_labels,mode='train')
train_mask, val_mask,_,_ = train_test_split_fn(train_attn,pr_train,mode='train')


train_inputs =torch.LongTensor(train_inputs)
train_labels = torch.LongTensor(train_labels)
train_mask = torch.LongTensor(train_mask)

val_inputs = torch.LongTensor(val_inputs)
val_labels = torch.LongTensor(val_labels)
val_mask = torch.LongTensor(val_mask)

test_inputs = torch.tensor(test_input)
test_labels = torch.tensor(test_labels)
test_mask = torch.tensor(test_mask)



train_data = TensorDataset(train_inputs, train_mask, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler= train_sampler, batch_size= args.train_batch_size)

val_data = TensorDataset(val_inputs, val_mask, val_labels)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler= val_sampler, batch_size=args.train_batch_size)

test_data = TensorDataset(test_inputs, test_mask, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler,batch_size=args.test_batch_size)
