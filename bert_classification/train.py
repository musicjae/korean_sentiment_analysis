# cf: https://medium.com/@eyfydsyd97/bert-for-question-answer-fine-tuning%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-by-pytorch-fbe15fdef330

import torch

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import random
import time

from bert_classification.arg import args
from bert_classification.model import bert, device
from bert_classification.metrics import flat_accuracy,optimizer,scheduler, format_time
from bert_classification.bert_data import train_dataloader, val_dataloader, test_dataloader

PATH = r'C:\Users\USER\Desktop\gitgit\korean_sentiment_analysis\bert_classification'
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

for epoch_i in range(0,args.epochs):

    bert.train()
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0

    bert.train()
    for step, batch in enumerate(train_dataloader):

        bert.zero_grad()

        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        batch= tuple(t.to(device) for t in batch)
        b_input_ids,b_input_mask,b_labels = batch

        ## ======= Foward ====== ##
        outputs = bert(b_input_ids,
                       token_type_ids= None,
                       attention_mask = b_input_mask,
                       labels = b_labels)

        loss = outputs[0] # the first returned element is the Cross Entropy loss between the predictions and the passed labels.
        total_loss += loss.item()
        loss.backward()
        """
        Gradient Clipping
          
          * 출력 길이에 따라 기울기 달라짐 --> 길이 길면 기울기 커짐 --> 매번 최적의 학습률을 찾아 조절하기 어려움 
          --> 그래디언트 클리핑 사용 --> net의 param의 norm(L2)를 구한 뒤 크기 제한 
          --> 제한된 최대 크기보다 norm이 큰 경우 grad가 클리핑됨 --> 능동적 학습율 조절이 가능해짐
        """
        torch.nn.utils.clip_grad_norm_(bert.parameters(),1.0) # Set max_norm = 1

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))


    ###################################################
    ####### =========== Validation ============ #######
    ###################################################

    t0 = time.time()
    print("Running Validation...")
    bert.eval()

    eval_loss, eval_accuracy = 0,0
    nb_eval_steps, nb_eval_examples = 0,0

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask,b_labels = batch

        with torch.no_grad(): # eval 에서는 grad 계산을 하지 않고 오직 검증만 한다
            outputs = bert(b_input_ids,
                           token_type_ids=None,
                           attention_mask = b_input_mask) # token_type_ids: 문장 유형을 구분한다. # attention_mask: 문장인 부분에 1을 PAD부분에는 0을 넣어준다

        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

torch.save(bert.state_dict(),PATH)



