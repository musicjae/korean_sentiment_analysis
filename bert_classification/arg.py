import argparse

parser = argparse.ArgumentParser(description='hyperparameters.....')
args = parser.parse_args("")

# =========== Training ============ #

args.train_batch_size = 32
args.test_batch_size = 32
args.epochs = 2
args.lr = 2e-5
args.eps = 1e-8
args.total_steps = 1055*args.epochs # 총 훈련 스텝 =  배치반복 횟수 * 에폭 where 1055 is len(train_loader)
                                    # To avoid the error 'cannot import name 'args' from partially initialized module 'bert_classification.arg' (most likely due to a circular import)'