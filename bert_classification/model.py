import torch

from transformers import BertForSequenceClassification


# 디바이스 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# 분류를 위한 BERT 모델 생성
bert = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
bert.cuda()