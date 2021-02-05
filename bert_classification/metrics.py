import numpy as np
from transformers import  AdamW
from transformers import get_linear_schedule_with_warmup
from bert_classification.arg import args
from bert_classification.model import bert
import datetime


def flat_accuracy(preds,labels):

    pred_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 옵티마이저 설정
optimizer = AdamW(bert.parameters(),
                  lr = args.lr, # 학습률
                  eps = args.eps # 0으로 나누는 것을 방지하기 위한 epsilon 값
                )


# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = args.total_steps)


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))