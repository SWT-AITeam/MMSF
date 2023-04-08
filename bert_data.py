import os

from encoder.bert_encoder import BERTEncoder
from models.rifre_sentence import RIFRE_SEN
from models.rifre_triple import RIFRE_TR
from framework.sentence_re import Sentence_RE
from framework.triple_re import Triple_RE
from configs import Config
from utils import count_params
import numpy as np
import torch
import random, argparse
torch.cuda.set_device(3)

# os.environ["CUDA_VISIBLE_DEVICES"]='3'

def seed_torch(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--train', default=True, type=bool)

    args = parser.parse_args()
    dataset = args.dataset
    is_train = args.train
    config = Config()
    if config.seed is not None:
        print(config.seed)
        seed_torch(config.seed)


    print('train--' + dataset)
    config.class_nums = config.veh_class
    sentence_encoder = BERTEncoder('bert-base-chinese')
    model = RIFRE_SEN(sentence_encoder, config)
    count_params(model)
    framework = Sentence_RE(model,
                            train_path=config.train,
                            val_path=config.val,
                            test_path=config.test,
                            rel2id=config.rel2id,
                            pretrain_path=config.bert_base,
                            ckpt=config.semeval_ckpt,
                            batch_size=config.batch_size,
                            max_epoch=config.epoch,
                            lr=config.lr)

    framework.train_semeval_model()
    # framework.train_model()
    framework.load_state_dict(config.semeval_ckpt)
    print('test:')
    framework.eval_semeval(framework.test_loader)
