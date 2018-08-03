# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     model
   Description : NER模型层
   Author :       charl
   date：          2018/7/31
-------------------------------------------------
   Change Activity:
                   2018/7/31:
-------------------------------------------------
"""

import numpy as np
import tensorflow as tf

from tensorflow.contrib.crf import crf_log_likelihood # 在一个条件随机场里面计算一个标签序列的log-likelihood
from tensorflow.contrib.crf import viterbi_decode # 通俗来说就是返回最好的标签序列，这个函数只能在测试时用，在tf外部解码
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.rnn import LSTMCell

from utils import result_to_json
from data_utils import create_input, iobes_iob
from utils import get_logger

class BILSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        '''
        构建图
        :return:
        '''
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_layer_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholers(self):
        '''
        构建输入
        :return:
        '''
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None, None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        # 这一步的作用
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,dtype=tf.float32,
                                           trainable=self.update_embedding, name="_word_embeddings")

            # embedding_lookup 就是根据inpus_ids中的id，来寻找embeddings中id行。比如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，组成一个tensor返回。
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell



