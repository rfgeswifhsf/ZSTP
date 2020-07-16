# -*- coding:utf-8 -*-
from utils import tokenize,build_vocab,read_vocab
import tensorflow as tf
from model import NerModel
import tensorflow_addons as tf_ad
import os
import numpy as np
from args_help import args
from my_log import logger



if not (os.path.exists(args.vocab_file) and os.path.exists(args.tag_file)):
    logger.info("building vocab file")
    build_vocab([args.train_path], args.vocab_file, args.tag_file)
else:
    logger.info("vocab file exits!!")


vocab2id, id2vocab = read_vocab(args.vocab_file)

tag2id, id2tag = read_vocab(args.tag_file)

text_sequences ,label_sequences= tokenize(args.train_path,vocab2id,tag2id)


train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences))
print(type(train_dataset))
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)
