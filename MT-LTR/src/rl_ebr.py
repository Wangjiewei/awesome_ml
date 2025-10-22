#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description:
    * state: query[, session, user]
    * action: poi list
    * reward: click + ndcg(relevance)
    * model: reinforce + twinbert + PER
author: clzhang
date: 2022/10/13
"""

from difflib import restore
import json
from tkinter import N
from tkinter.messagebox import NO
from turtle import st
#from multi_gpu_twbrt3.inputs import DataProcessor
import numpy as np
from collections import deque, namedtuple, defaultdict, OrderedDict, defaultdict
import tensorflow.compat.v1 as tf
import tensorflow as tf_2
import mkl
import time
import os
import gc
import random
import faiss
import logging
from faiss import METRIC_INNER_PRODUCT, normalize_L2
from sklearn.preprocessing import normalize
#distances = cosine_distances(eq, epoi)
from sklearn.metrics.pairwise import cosine_distances

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tokenization 
import modeling_rl_woln as modeling
from modeling_rl_woln import create_initializer
mkl.get_max_threads()
tf.disable_eager_execution() # 执行错误就restart jupyter kernel
#tf.config.experimental_run_functions_eagerly(True)

BState = namedtuple("BatchState", field_names=["queries", "sessions", "users", "labels"])
BAction = namedtuple("BatchAction", field_names=["dists", "poiids", 'poi_infos', 'labels'])
flags = tf.flags
FLAGS = flags.FLAGS
## I/O parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string("task_name", 'regression', "The name of the task to train.")
flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("vector_file", None, "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("test_vector_file", None, "The output directory where the model checkpoints will be written.")
## processing data parameters
flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")
flags.DEFINE_integer(
    "max_seq_length", 72,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string("train_tf_record_file", '../samples/BR_20230301_20230331_rl_train_data_tfrecord_br/part-*', "The train file that the BERT model was trained on.")
flags.DEFINE_string("test_tf_record_file", '../samples/BR_20230401_20230410_rl_train_data_tfrecord_br/part-*', "The test file that the BERT model was tested on.")
flags.DEFINE_string("test_tf_record_file_2", '../samples/BR_20230401_20230410_rl_train_data_tfrecord_br/part-*', "The test file that the BERT model was tested on.")
flags.DEFINE_string("info_file", "test_poi.tsv", "predict vector file name")

flags.DEFINE_integer("query_type_class_num", 2, "number of query_type.") #new
flags.DEFINE_integer("geohash_class_num", 540899, "number of geohash_ids.")
flags.DEFINE_integer("area_class_num", 554, "number of area_ids.")
flags.DEFINE_integer("category_class_num", 453, "number of category_ids.")
flags.DEFINE_integer("week_bucket_class_num", 8, "number of day_bucket.")
flags.DEFINE_integer("day_bucket_class_num", 7, "number of week_bucket.")
flags.DEFINE_integer("hidden_size", 128, "feature's hidden embedding size.")


# gpu parameters
# training parameters
flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")
flags.DEFINE_integer("search_batch_size", 64, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 128, "Total batch size for predict.")
flags.DEFINE_integer("index_size", 128, "index hidden size.")
flags.DEFINE_float("learning_rate", 3e-7, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_train_epochs", 300, "Total number of training epochs to perform.")
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "How often to save the model checkpoint.")
flags.DEFINE_integer("keep_checkpoint_max", 3, "How many checkpoint file will be saved")
flags.DEFINE_string("gpu_ids", "1", "predict vector file name")
#flags.DEFINE_integer("replybuffer_size", 10000, "replybuffer size")
#flags.DEFINE_integer("update_steps", 6000, "update steps")
flags.DEFINE_integer("replybuffer_size", 6000, "replybuffer size")
flags.DEFINE_integer("update_steps", 6000, "update steps")
flags.DEFINE_integer("random_sampe_size", 5, "sample size from replybuffer ")
flags.DEFINE_integer("reward_queue_size", 10, "")
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_ids

def convert_token_ids(example, max_seq_len, tokenizer):
    '''本地生产query tokenid'''
    tokens = tokenizer.tokenize(example)
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[0:(max_seq_len - 2)]
    tokens = ["[CLS]"] + tokens
    tokens.append("[SEP]")
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_len = len(token_ids)
    segment_ids = [0]*max_seq_len
    mask_ids = [1] * token_len
    if token_len < max_seq_len:
        delta_len = max_seq_len - token_len
        token_ids.extend([0]*delta_len)
        mask_ids.extend([0]*delta_len)
    else:
        token_ids = token_ids[:max_seq_len]
        segment_ids = segment_ids[:max_seq_len]
        mask_ids = mask_ids[:max_seq_len]
    return token_ids, segment_ids, mask_ids

class EBRNetworkNew(object):

    def __init__(self, bert_config, max_seq_len=32, clip_coef=0.2):
        self.bert_config = bert_config
        self.query_input_ids = tf.placeholder(tf.int64, [None,max_seq_len])
        self.query_input_mask = tf.placeholder(tf.int64, [None, max_seq_len])
        self.query_segment_ids = tf.placeholder(tf.int64, [None, max_seq_len])
        self.bert_dropout_ratio = tf.placeholder(tf.float32)
        self.bert_output_dropout_ratio = tf.placeholder(tf.float32)
        self.pos_poi_input_ids = tf.placeholder(tf.int64, [None,max_seq_len])
        self.pos_poi_input_mask = tf.placeholder(tf.int64, [None, max_seq_len])
        self.pos_poi_segment_ids = tf.placeholder(tf.int64, [None, max_seq_len])
        self.neg_poi_input_ids = tf.placeholder(tf.int64, [None,max_seq_len])
        self.neg_poi_input_mask = tf.placeholder(tf.int64, [None, max_seq_len])
        self.neg_poi_segment_ids = tf.placeholder(tf.int64, [None, max_seq_len])
        self.poi_cos_dists = tf.placeholder(tf.float32, [None, 1])
        self.rewards = tf.placeholder(tf.float32, [None, 1])

        # new poi features:
        self.pos_poi_area_id = tf.placeholder(tf.int64, [None, ])
        self.pos_poi_geocode_id = tf.placeholder(tf.int64, [None, ])
        self.pos_poi_click_score = tf.placeholder(tf.float32, [None,])
        self.pos_poi_category_ids = tf.placeholder(tf.int64, [None, 4])
        self.pos_poi_category_mask = tf.placeholder(tf.int64, [None, 4])
        self.neg_poi_area_id = tf.placeholder(tf.int64, [None, ])
        self.neg_poi_geocode_id = tf.placeholder(tf.int64, [None, ])
        self.neg_poi_click_score = tf.placeholder(tf.float32, [None, ])
        self.neg_poi_category_ids = tf.placeholder(tf.int64, [None, 4])
        self.neg_poi_category_mask = tf.placeholder(tf.int64, [None, 4])

        # new query features:
        self.query_type = tf.placeholder(tf.int64, [None, 1])
        self.query_city_id = tf.placeholder(tf.int64, [None, 1])
        self.query_geohash_id = tf.placeholder(tf.int64, [None, 1])
        self.query_week = tf.placeholder(tf.int64, [None, 1])
        self.query_day = tf.placeholder(tf.int64, [None, 1])


        self.init_embedding_table()
        self.pos_poi_click_score_tensor, self.pos_geohash_tensor, self.pos_category_tensor = self.get_poi_side_feature(self.pos_poi_area_id, self.pos_poi_geocode_id, self.pos_poi_click_score, self.pos_poi_category_ids, self.pos_poi_category_mask)
        self.neg_poi_click_score_tensor, self.neg_geohash_tensor, self.neg_category_tensor = self.get_poi_side_feature(self.neg_poi_area_id, self.neg_poi_geocode_id, self.neg_poi_click_score, self.neg_poi_category_ids, self.neg_poi_category_mask)
        self.query_type_tensor, self.query_geohash_tensor, self.query_week_bkt_tensor, self.query_day_bkt_tensor = self.get_query_side_feature(self.query_type, self.query_geohash_id, self.query_week, self.query_day)

        self.query_pooling_token_tensor_text = self.get_bert_out(self.query_input_ids, self.query_input_mask, self.query_segment_ids)
        self.pos_pooling_token_tensor_text = self.get_bert_out(self.pos_poi_input_ids, self.pos_poi_input_mask, self.pos_poi_segment_ids)
        self.neg_pooling_token_tensor_text = self.get_bert_out(self.neg_poi_input_ids, self.neg_poi_input_mask, self.neg_poi_segment_ids)

        self.query_pooling_token_tensor_text = tf.expand_dims(self.query_pooling_token_tensor_text, axis=1) # 
        self.pos_pooling_token_tensor_text = tf.expand_dims(self.pos_pooling_token_tensor_text, axis=1)
        self.neg_pooling_token_tensor_text = tf.expand_dims(self.neg_pooling_token_tensor_text, axis=1)


        self.query_pooling_token_tensor = tf.reduce_mean(tf.concat( \
                                            [self.query_pooling_token_tensor_text, self.query_type_tensor,  self.query_week_bkt_tensor, self.query_day_bkt_tensor, self.query_geohash_tensor], \
                                           axis=1), axis=1)
        self.pos_pooling_token_tensor = tf.reduce_mean(tf.concat( \
                                            [self.pos_pooling_token_tensor_text, self.pos_poi_click_score_tensor, self.pos_category_tensor, self.pos_geohash_tensor], \
                                            axis=1), axis=1)
        self.neg_pooling_token_tensor = tf.reduce_mean(tf.concat( \
                                            [self.neg_pooling_token_tensor_text, self.neg_poi_click_score_tensor,  self.neg_category_tensor, self.neg_geohash_tensor], \
                                            axis=1), axis=1)

        #self.query_pooling_token_tensor = self.get_bert_out(self.query_input_ids, self.query_input_mask, self.query_segment_ids)
        #self.pos_pooling_token_tensor = self.get_bert_out(self.pos_poi_input_ids, self.pos_poi_input_mask, self.pos_poi_segment_ids)
        #self.neg_pooling_token_tensor = self.get_bert_out(self.neg_poi_input_ids, self.neg_poi_input_mask, self.neg_poi_segment_ids)

        with tf.variable_scope("crossing"):
            constant_zero = tf.constant(0.0, dtype=tf.float32)   # tensorflow 2.x  return negitive number
            cosin_similarity_qp = constant_zero - tf.keras.losses.cosine_similarity(self.query_pooling_token_tensor, \
                                                                    self.pos_pooling_token_tensor, axis=1)
            cosin_similarity_qp = tf.reshape(cosin_similarity_qp, [-1, 1])
            cosin_similarity_qn = constant_zero - tf.keras.losses.cosine_similarity(self.query_pooling_token_tensor, \
                                                                    self.neg_pooling_token_tensor, axis=1)
            cosin_similarity_qn = tf.reshape(cosin_similarity_qn, [-1, 1])

        with tf.variable_scope("loss"):
            constant_one = tf.constant(1.0, dtype=tf.float32)
            distance_qp = constant_one - cosin_similarity_qp
            distance_qn = constant_one - cosin_similarity_qn
            margin = tf.constant(0.2, dtype=tf.float32)
            per_example_loss = tf.maximum(distance_qp - distance_qn + margin, 0)
            self.loss = tf.reduce_mean(per_example_loss)

        weights_var = tf.trainable_variables()
        grads = tf.gradients(self.loss, weights_var)
        optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate,
        epsilon=1e-6)
        self.train_op = optimizer.apply_gradients(zip(grads, weights_var))

    def init_embedding_table(self):
        with tf.variable_scope("embedding_tables", reuse=tf.AUTO_REUSE):
            self.query_type_embedding_table = tf.get_variable(
                name='query_type',
                shape=[FLAGS.query_type_class_num, FLAGS.hidden_size],
                initializer=create_initializer(initializer_range=0.02)) #query

            self.geohash_embedding_table = tf.get_variable(
                name='geohash',
                shape=[FLAGS.geohash_class_num, FLAGS.hidden_size],
                initializer=create_initializer(initializer_range=0.02)) #poi/query

            self.area_embedding_table = tf.get_variable(
                name='area',
                shape=[FLAGS.area_class_num, FLAGS.hidden_size],
                initializer=create_initializer(initializer_range=0.02)) #

            self.category_embedding_table = tf.get_variable(
                name='category',
                shape=[FLAGS.category_class_num, FLAGS.hidden_size],
                initializer=create_initializer(initializer_range=0.02)) #poi

            self.week_bkt_embedding_table = tf.get_variable(
                name='week_bucket',
                shape=[FLAGS.week_bucket_class_num, FLAGS.hidden_size],
                initializer=create_initializer(initializer_range=0.02)) #query

            self.day_bkt_embedding_table = tf.get_variable(
                name='day_bucket',
                shape=[FLAGS.day_bucket_class_num, FLAGS.hidden_size],
                initializer=create_initializer(initializer_range=0.02)) #query

            self.poi_click_score_embedding = tf.get_variable(
                name='click_score',
                shape=[1, FLAGS.hidden_size],
                initializer=create_initializer(initializer_range=0.02)) #poi

    def get_poi_side_feature(self, poi_area_id, poi_geocode_id, poi_click_score, poi_category_ids, poi_category_mask ):
        poi_category_mask = tf.to_float(poi_category_mask)

        poi_click_score = tf.expand_dims(poi_click_score, axis=1)
        poi_click_score_tensor = poi_click_score * self.poi_click_score_embedding

        # pos_area_tensor = tf.gather(area_embedding_table, pos_poi_city_id)
        geohash_tensor = tf.gather(self.geohash_embedding_table, poi_geocode_id)
        category_tensor = tf.gather(self.category_embedding_table, poi_category_ids)

        poi_category_mask = tf.expand_dims(poi_category_mask, axis=-1)
        category_tensor = tf.reduce_sum(category_tensor * poi_category_mask, axis=1)

        poi_click_score_tensor = tf.expand_dims(poi_click_score_tensor, axis=1)
        # pos_area_tensor = tf.expand_dims(pos_area_tensor, axis=1)
        geohash_tensor = tf.expand_dims(geohash_tensor, axis=1)
        category_tensor = tf.expand_dims(category_tensor, axis=1)
        
        return poi_click_score_tensor,  geohash_tensor, category_tensor

    def get_query_side_feature(self, query_type, query_geohash_id, query_week, query_day):
        query_type_tensor = tf.gather(self.query_type_embedding_table, query_type)
        query_geohash_tensor = tf.gather(self.geohash_embedding_table, query_geohash_id)
        # query_city_tensor = tf.gather(area_embedding_table, query_city_id)
        query_week_bkt_tensor = tf.gather(self.week_bkt_embedding_table, query_week)
        query_day_bkt_tensor = tf.gather(self.day_bkt_embedding_table, query_day)
        
        ##query_type_tensor = tf.expand_dims(query_type_tensor, axis=1)
        ##query_geohash_tensor = tf.expand_dims(query_geohash_tensor, axis=1)
        # query_city_tensor = tf.expand_dims(query_city_tensor, axis=1)
        ##query_week_bkt_tensor = tf.expand_dims(query_week_bkt_tensor, axis=1)
        ##query_day_bkt_tensor = tf.expand_dims(query_day_bkt_tensor, axis=1)

        return query_type_tensor, query_geohash_tensor, query_week_bkt_tensor, query_day_bkt_tensor


    def get_bert_out(self, input_ids, input_mask, segment_ids):
        bert_tower_model = modeling.BertModel(
            config=self.bert_config,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False,
            comp_type=tf.float32,
            scope="bert",
            bert_dropout=self.bert_dropout_ratio)

        bert_output_layer = bert_tower_model.get_sequence_output()

        
        bert_embedding = bert_tower_model.get_embedding_output()

        with tf.variable_scope("pooling", reuse=tf.AUTO_REUSE):
            bert_index_output = tf.layers.dense(
                bert_output_layer,
                FLAGS.index_size,
                kernel_initializer=modeling.create_initializer(0.02))
            bert_idx_dropout_ratio = self.bert_output_dropout_ratio
            bert_output_layer = modeling.dropout(bert_index_output, bert_idx_dropout_ratio)
            bert_output_layer = modeling.layer_norm(bert_output_layer)

            hidden_size = bert_output_layer.shape[-1]
            bert_weights = tf.get_variable(
                "bert_weights", [1, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            bert_bias = tf.get_variable(
                "bert_bias", [1], initializer=tf.zeros_initializer())

            bert_output_layer = bert_output_layer[:, 1:, :]
            bert_seq_layer = tf.squeeze(tf.nn.bias_add(tf.matmul(bert_output_layer, bert_weights, transpose_b=True), \
                                                    bert_bias), axis=2)
            bert_attention_score = tf_2.nn.softmax(bert_seq_layer, axis=-1)
            bert_pooling_token_tensor = tf.reduce_sum(tf.expand_dims(bert_attention_score, -1) * bert_output_layer, axis=1)

        return bert_pooling_token_tensor

class PoiHNSW(object):
    '''HNSW 模块'''
    def __init__(self, index_dir="fasis_indexs"):
        self.index_dir = index_dir

    def train_index(self, epoi, index_type='hnsw', d=128, poi_idx=None):
        '''
        生成离线索引的过程
        index_type: 索引的类型，当前模块支持了hnsw、pq压缩后的倒排索引、sqfp sq压缩后索引，flatindex暴力索引
        poi_idx: faiss 建索引时默认使用向量的顺序编号作为id的，如果要增加映射，可以增加idx对应的id list
        '''
        epoi = np.array(epoi).astype(np.float32)
        normalize_L2(epoi) # 归一化之后的内积 就是cos
        #epoi = epoi.astype(np.float16) # faiss 不支持
        if index_type == 'hnsw':
            # hnsw
            M = 20  # the number of neighbors used in the graph.
            index = faiss.IndexHNSWFlat(d, M, METRIC_INNER_PRODUCT)  # , efConstruction=10, efSearch=4)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 600
            #index = faiss.index_cpu_to_all_gpus(index)
            print("hnsw")
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(quantizer, d, 10000, 8, 8)
            index = faiss.index_cpu_to_all_gpus(index)
        elif index_type == "sqfp":
            index = faiss.index_factory(d, 'SQfp16')
            index = faiss.index_cpu_to_all_gpus(index)
        else:
            # flat with cosine (Exact Search)
            index = faiss.IndexFlatIP(d)
            index = faiss.index_cpu_to_all_gpus(index)
        
        index.train(epoi)
        index = faiss.IndexIDMap(index)
        if poi_idx:
            poi_idx = np.array(poi_idx)
            index.add_with_ids(epoi, poi_idx)
        else:
            index.add(epoi)
        return index

    def index_add(self, epoi, index):
        '''做增量索引时使用'''
        epoi = np.array(epoi).astype(np.float32)
        normalize_L2(epoi)
        #epoi = epoi.astype(np.float16)
        index.add(epoi)

    def _read_vec(self, line):
        '''解析token文件'''
        line = line.strip().split("\t")
        poiid, city, token_ids,area_id, poi_geocode_id, click_score, poi_category_ids, poi_category_mask = line[:8]
        #return poiid, city, [int(i) for i in token_ids.split(";")]
        return int(poiid), int(city), token_ids, int(area_id), int(poi_geocode_id), float(click_score), poi_category_ids, poi_category_mask

    def get_all_poi_tokens(self, poi_vec_path):
        '''读取初始化token 文件'''
        all_token_ids = []
        index = None
        # poi_idx = {}
        poi_city_idx = {}
        all_idxs = []
        with open(poi_vec_path, 'r') as f:
            idx = 0
            for line in f:
                poiid, city, token_ids, area_id, poi_geocode_id, click_score, poi_category_ids, poi_category_mask = self._read_vec(line)
                #if city != 55000360:   ## to be delete
                #    continue
                if city not in poi_city_idx:
                    poi_city_idx[city] =[]
                all_token_ids.append((city, token_ids, area_id, poi_geocode_id, click_score, poi_category_ids, poi_category_mask))
                poi_city_idx[city].append(idx)
                all_idxs.append(idx)
                idx += 1
                # test few pois
                #if idx > 1000000:
                #    break
        return all_token_ids, poi_city_idx, all_idxs

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["query", "label", "reward", "poiid", "pos_tokens"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, info):
        """Add a new experience to memory."""
        # 每个query召回的poi 逐个存入replybuffer内，方便shuffer
        for idx, token_ids in enumerate(state.queries["token_ids"]):
            pois = action.poiids[idx]
            pos_token = state.labels["pos_token_ids"][idx].tolist()
            # 如果存在padding id 剔除掉，节省内存
            zero_pos = len(pos_token)
            if 0 in pos_token:
                zero_pos = pos_token.index(0)
            pos_token = pos_token[:zero_pos]
            for poi in pois:
                #print(state.queries)
                #print(state.labels)
                #print(reward)
                e = self.experience({"token_ids": token_ids, "segment_ids": state.queries["segment_ids"][idx], "mask_ids": state.queries["mask_ids"][idx],"query_type": state.queries["query_type"][idx], "query_city_id":state.queries["query_city_id"][idx], "query_geohash_id":state.queries["query_geohash_id"][idx],"query_week": state.queries["query_week"][idx],"query_day": state.queries["query_day"][idx]}, state.labels["labels"][idx][0], reward[idx], poi, pos_token)
                self.memory.append(e)


    def sample(self, sample_num):
        """Randomly sample a batch of experiences from memory."""
        #print("deque:", len(self.memory), sample_num)
        experiences = random.sample(self.memory, k=sample_num)  # add时逐个存k=self.batch_size
        return experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent(object):
    def __init__(self, buffer_size, batch_size, update_steps, top_k=100):
        self.optimizer = tf.keras.optimizers.Adam()
        self.index = None
        self.test_index = None
        self.batch_size = batch_size
        self.rb = ReplayBuffer(buffer_size, self.batch_size, seed=0)
        self.t_step = 0
        self.all_steps = 0
        self.update_steps = update_steps
        self.batch_dist_ratios = None
        self.network = EBRNetworkNew(bert_config)
        self.saver = tf.train.Saver()
        self.top_k = top_k
        self.sess = self._init_session()
        if(FLAGS.init_checkpoint is not None):
            #self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.init_checkpoint))
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
            tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.poihnsw = PoiHNSW()
        start_time = int(time.time())
        self.poi_token_ids, self.poi_city_idx, self.all_idx = self.poihnsw.get_all_poi_tokens(FLAGS.vector_file)
        # self.test_token_ids, _, _ = self.poihnsw.get_all_poi_tokens(FLAGS.test_vector_file)
        self.restruct_all_index(is_test=False, poi_token_ids=self.poi_token_ids)
        # self.restruct_all_index(is_test=True, poi_token_ids=self.test_token_ids)
        print(len(self.poi_token_ids))
        print("hnsw start time:", int(time.time()) - start_time)
        self.top5_ratio = 0.0

    def _init_session(self):
        configure = tf.ConfigProto()
        configure.gpu_options.allow_growth=True
        configure.gpu_options.per_process_gpu_memory_fraction = 1
        configure.log_device_placement = False
        return tf.Session(config=configure)

    def get_city_query(self, query_embed, labels, queries):
        city_query = defaultdict(list)
        city_label = defaultdict(list)
        city_idx_index = defaultdict(list)
        raw_labels = labels["labels"].tolist()
        citys = labels["city_id"].tolist()
        for idx, embed in enumerate(query_embed):
            city = citys[idx][0]
            city_query[city].append(embed)
            city_label[city].append(raw_labels[idx][0])
            city_idx_index[city].append(idx)
        return city_query, city_label, city_idx_index 

    def act(self, state):
        '''
        state: query, session, user
        output action: poi list by bert model + hnsw index
        '''
        query_embs = self._create_token_pooling_embedding(state.queries)
        # 把query 按城市分开，分城市检索
        city_query, city_label, city_idx_index = self.get_city_query(query_embs, state.labels, state.queries)
        batch_size = len(query_embs)
        all_poiids = [0] * batch_size  ## TBD
        all_labels = [0] * batch_size
        total_poiids = []
        for city, embed in city_query.items():
            #if(city != 55000360): continue  ## delete
            dists, poiids, labels, _ = self._topk_pois_by_hnsw(embed, city_label[city], self.top_k, city=city)
            for idx, poiid in enumerate(poiids):
                total_poiids.append(poiid[:7] + poiid[13:]) 
                all_poiids[city_idx_index[city][idx]] = poiid[7:13]
                all_labels[city_idx_index[city][idx]] = labels[idx]
        all_poiids = [hard_poi + total_poiids[i] for i, hard_poi in enumerate(all_poiids)]
        action = BAction(dists, all_poiids, None, all_labels)  # poiids: [bs,k]
        return action

    def convert_tensor_to_np(self, one_batch_data):
        token_ids, segment_ids, mask_ids, city_id, pos_token_ids, query_type, query_city_id, query_geohash_id, query_week, query_day, labels = self.sess.run(
            [one_batch_data[0]["token_ids"], one_batch_data[0]["segment_ids"], one_batch_data[0]["mask_ids"],
              one_batch_data[0]["city_id"], one_batch_data[0]["poi_token_ids"], one_batch_data[0]["query_type"],one_batch_data[0]["query_city_id"], one_batch_data[0]["query_geohash_id"], one_batch_data[0]["query_week"], one_batch_data[0]["query_day"] , one_batch_data[1]])   # new
        queries = {"token_ids":token_ids, "segment_ids": segment_ids, "mask_ids": mask_ids, "query_type": query_type, "query_city_id":query_city_id, "query_geohash_id":query_geohash_id, "query_week":query_week, "query_day":query_day}
        return queries, {"labels": labels, "city_id": city_id, "pos_token_ids": pos_token_ids}

    def calc_top_ratio(self, click_ratios):
        '''
        click_ratios: [top1_num, top3_num, top5_num, top10_num, total_num]
        '''
        total = float(click_ratios[-1])
        top1 = round(click_ratios[0]/ total, 6)
        top3 = round(click_ratios[1]/ total, 6)
        top5 = round(click_ratios[2]/ total, 6)
        top10 = round(click_ratios[3]/ total, 6)
        return (top1, top3, top5, top10)

    def eval(self, data):
        '''
        data: 测试数据，dataset
        '''    
        all_topk_click_ratios = [0,0,0,0, 0]
        topk_click_ratios = [1,1,1,1,1]
        data_iter = data.make_one_shot_iterator()
        one_batch_data = data_iter.get_next()
        is_run = True
        start_time = int(time.time())
        while is_run:
            try:
                queries,labels = self.convert_tensor_to_np(one_batch_data)
                query_embeds = self._create_token_pooling_embedding(queries)
                city_query, city_label,_ = self.get_city_query(query_embeds, labels, queries)
                for city, embeds in city_query.items():
                    #if(city != 55000360): continue  ## to be delete
                    #print("eval_city:{}, eval_label:{}".format( city, city_label[city]))
                    _, _, _, topk_click_ratios = self._topk_pois_by_hnsw(embeds, city_label[city], self.top_k, is_test=False, city=city)
                all_topk_click_ratios = [i + j for i, j in zip(all_topk_click_ratios, topk_click_ratios)]
            except tf.errors.OutOfRangeError:
                is_run = False
        print("eval_time:", int(time.time()) - start_time)
        topk_ratio = self.calc_top_ratio(all_topk_click_ratios)
        print("eval top1:%s\ttop3:%s\ttop5:%s\ttop10:%s\n" % topk_ratio)
        return topk_ratio

    def _create_token_pooling_embedding(self, queries):  # query final embedding
        # 生产qery向量  
        query_embs = self.sess.run([self.network.query_pooling_token_tensor], feed_dict={
                 self.network.query_input_ids: queries["token_ids"],
                 self.network.query_input_mask: queries["mask_ids"],
                 self.network.query_segment_ids: queries["segment_ids"],
                self.network.query_type: queries["query_type"],
                self.network.query_city_id: queries["query_city_id"],
                self.network.query_geohash_id: queries["query_geohash_id"],
                self.network.query_week: queries["query_week"],
                self.network.query_day: queries["query_day"],
                self.network.bert_dropout_ratio: 0.0,
                self.network.bert_output_dropout_ratio: 0.0
             })
        return query_embs[0]

    def _create_poi_token_pooling_embedding(self, pois):  # query final embedding
        # 生产qery向量
        #print("{},{},{},{},{}".format(pois["area_id"].shape, pois["geocode_id"].shape, pois["click_score"].shape))
        poi_embs = self.sess.run([self.network.pos_pooling_token_tensor], feed_dict={
                 self.network.pos_poi_input_ids: pois["token_ids"],
                 self.network.pos_poi_input_mask: pois["mask_ids"],
                 self.network.pos_poi_segment_ids: pois["segment_ids"],
                self.network.pos_poi_area_id: pois["area_id"],
                self.network.pos_poi_geocode_id: pois["geocode_id"],
                self.network.pos_poi_click_score: pois["click_score"],
                self.network.pos_poi_category_ids: pois["category_ids"],
                self.network.pos_poi_category_mask: pois["category_mask"],
                self.network.bert_dropout_ratio: 0.0,
                self.network.bert_output_dropout_ratio: 0.0        
             })
        return poi_embs[0]

    def _get_topk_click_ratio(self, pos, click_ratios):
        '''计算个sample的点击'''
        if pos == 0:
            click_ratios[0] += 1
        if pos <= 2:
            click_ratios[1] += 1
        if pos <= 4:
            click_ratios[2] += 1
        if pos <= 9:
            click_ratios[3] += 1
        click_ratios[4] += 1
        return click_ratios

    def _topk_pois_by_hnsw(self, query_embs, raw_labels, k=40, is_test=False, city=0):
        '''
        query_embs: query向量，[bs, emb_size]
        raw_labels: user click pois
        k: 检索条数
        '''
        query_embs = normalize(query_embs, axis=1, norm='l2')
        if city not in self.index:
            city = default_city
        if is_test:
            sim_distances, neighbors = self.test_index.search(query_embs.astype(np.float32), k+1)
        else:
            sim_distances, neighbors = self.index[city].search(query_embs.astype(np.float32), k+1)
        labels = []
        topk_click_ratio = [0, 0, 0, 0, 0]
        all_pois = []
        for idx, pois in enumerate(neighbors):
            #print(len(pois))
            click_poi = raw_labels[idx]
            pois = pois.tolist()
            
            r_pos = k+1
            label = [0, 1,2,3,4,5,6,7,8,9]
            if click_poi in pois:
                r_pos = pois.index(click_poi)
                if r_pos < 10:
                    label[r_pos] = 10 
                if r_pos >= 10:
                    r_pos = 10
                pois.remove(click_poi)
            all_pois.append(pois)
            labels.append(label)
            topk_click_ratio = self._get_topk_click_ratio(r_pos, topk_click_ratio)
        return sim_distances, all_pois, labels, topk_click_ratio

    def restruct_all_index(self, is_test=False, poi_token_ids=[]):
        info_size = FLAGS.predict_batch_size
        start_time = int(time.time())
        if is_test:
            del self.test_index
            self.test_index = {}
        else:
            del self.index
            self.index = {}
        gc.collect()
        print("start restruct")
        start_time = int(time.time())
        total = 0
        #info_size=100 
        # 在每个城市内抽取部分poi建立索引
        for city, all_city_idx in self.poi_city_idx.items():
            length = len(all_city_idx)
            # 随机采样一定比例的poi 建立索引 
            if city == default_city or length < 1000:
                city_idx = all_city_idx
            else:
                city_idx =  random.sample(all_city_idx, int(length/4))

            print("city:{} has {} pois to restruct".format(city, len(city_idx)))
            city_idx.sort()
            all_embeds = []
            batch_embeds = []
            batch_token_ids = []
            batch_segment_ids = []
            batch_mask_ids = []
            batch_area_id = [] # new
            batch_poi_geocode_id = []
            batch_click_score = []
            batch_poi_category_ids = []
            batch_poi_category_mask = []
            for i, poi_idx in enumerate(city_idx):
                new_token_ids, segment_ids, mask_ids = self.get_poi_inputs(poi_token_ids[poi_idx][1])
                area_id, poi_geocode_id, click_score, poi_category_ids, poi_category_mask = poi_token_ids[poi_idx][2:]  #新增poi特征
                poi_category_ids = poi_category_ids.split(',')
                poi_category_mask = poi_category_mask.split(',')
                batch_token_ids.append(new_token_ids) 
                batch_segment_ids.append(segment_ids)
                batch_mask_ids.append(mask_ids)
                batch_area_id.append(area_id) # new
                batch_poi_geocode_id.append(poi_geocode_id)
                batch_click_score.append(click_score)
                batch_poi_category_ids.append(poi_category_ids)
                batch_poi_category_mask.append(poi_category_mask)
                if i > 0 and len(batch_token_ids) % info_size == 0:
                    poi_vecs = self._create_poi_token_pooling_embedding(
                        {"token_ids": batch_token_ids, "segment_ids": batch_segment_ids, "mask_ids": batch_mask_ids, "area_id": batch_area_id, "geocode_id": batch_poi_geocode_id, "click_score": batch_click_score, "category_ids": batch_poi_category_ids, "category_mask": batch_poi_category_mask})
                    #print(len(poi_vecs))
                    batch_embeds.extend(poi_vecs.astype(np.float32).tolist())
                    batch_token_ids = []
                    batch_segment_ids = []
                    batch_mask_ids = []
                    batch_area_id = []
                    batch_poi_geocode_id = []
                    batch_click_score = []
                    batch_poi_category_ids = []
                    batch_poi_category_mask = []
                batch_len = len(batch_embeds)
                if batch_len > 0 and batch_len % 100000 == 0:
                    #print("all_len:", len(all_embeds))
                    if len(all_embeds) == 0:
                        all_embeds = np.array(batch_embeds).astype(np.float32)
                    else:
                        all_embeds = np.concatenate((all_embeds, np.array(batch_embeds).astype(np.float32)), axis=0)
                    batch_embeds = []
                total += 1
            if len(batch_token_ids) > 0:
                poi_vecs = self._create_poi_token_pooling_embedding({"token_ids": batch_token_ids, "segment_ids": batch_segment_ids, "mask_ids": batch_mask_ids,  "area_id": batch_area_id, "geocode_id": batch_poi_geocode_id, "click_score": batch_click_score, "category_ids": batch_poi_category_ids, "category_mask": batch_poi_category_mask})
                batch_embeds.extend(poi_vecs.astype(np.float32).tolist())
            if len(batch_embeds) > 0:
                if len(all_embeds) > 0 :
                    all_embeds = np.concatenate((all_embeds, np.array(batch_embeds).astype(np.float32)), axis=0)
                else:
                    all_embeds = np.array(batch_embeds).astype(np.float32)
            self.index[city] = self.poihnsw.train_index(all_embeds, poi_idx=city_idx)
            tf.logging.info("restruct: {}, \tuse time:{}".format( city,  int(time.time()) - start_time))
        print("restruct index total:", total, "\tcost:", int(time.time()) - start_time)

    def get_poi_inputs(self, token_id_str):
        if isinstance(token_id_str, str):
            token_ids = [int(i) for i in token_id_str.split(";")]
        else:
            token_ids = token_id_str
        #token_ids = token_ids.tolist()
        token_len = len(token_ids)
        name_len = token_len
        if sep_id in token_ids[1:]:
            name_len = token_ids[1:].index(sep_id)+2
        segment_ids = [0]*name_len + [1]*(token_len-name_len)
        mask_ids = [1]*token_len
        if token_len < FLAGS.max_seq_length:
            delta_len = FLAGS.max_seq_length-token_len
            padd_ids = [0]*delta_len
            token_ids.extend(padd_ids)
            segment_ids.extend(padd_ids)
            mask_ids.extend(padd_ids)
        return token_ids, segment_ids, mask_ids

    def _learn(self, experiences):
        # TODO: experiences convert to train samples
        # batch_train_samples, rewards = f(experiences)
        tf.logging.info("start_leran:")
        with tf.GradientTape() as tape:
            step_num = 0
            batch_poi_tokens = []
            batch_poi_segments = []
            batch_poi_masks = []
            batch_pos_poi_tokens = []
            batch_pos_poi_segments = []
            batch_pos_poi_masks = []
            batch_query_tokens = []
            batch_query_segments = []
            batch_query_masks = []
            #new
            batch_pos_poi_area_id = []
            batch_pos_poi_geocode_id = []
            batch_pos_poi_click_score = []
            batch_pos_poi_category_ids = []
            batch_pos_poi_category_mask = []
            batch_neg_poi_area_id = []
            batch_neg_poi_geocode_id = []
            batch_neg_poi_click_score = []
            batch_neg_poi_category_ids = []
            batch_neg_poi_category_mask = []
            batch_query_type = []
            batch_query_city_id = []
            batch_query_geohash_id = []
            batch_query_week = []
            batch_query_day = []
            new_batch_size = FLAGS.train_batch_size
            for e in experiences:
                if step_num != 0 and step_num % new_batch_size == 0:
                    loss, _ = self.sess.run([self.network.loss, self.network.train_op], feed_dict={
                            self.network.query_input_ids: batch_query_tokens,
                            self.network.query_input_mask: batch_query_masks,
                            self.network.query_segment_ids: batch_query_segments,
                            self.network.neg_poi_input_ids: batch_poi_tokens,
                            self.network.neg_poi_input_mask: batch_poi_masks,
                            self.network.neg_poi_segment_ids: batch_poi_segments,
                            self.network.pos_poi_input_ids: batch_pos_poi_tokens,
                            self.network.pos_poi_input_mask: batch_pos_poi_masks,
                            self.network.pos_poi_segment_ids: batch_pos_poi_segments,
                            self.network.pos_poi_area_id: batch_pos_poi_area_id,
                            self.network.pos_poi_geocode_id: batch_pos_poi_geocode_id,
                            self.network.pos_poi_click_score: batch_pos_poi_click_score,
                            self.network.pos_poi_category_ids: batch_pos_poi_category_ids,
                            self.network.pos_poi_category_mask: batch_pos_poi_category_mask,
                            self.network.neg_poi_area_id: batch_neg_poi_area_id,
                            self.network.neg_poi_geocode_id:batch_pos_poi_geocode_id,
                            self.network.neg_poi_click_score: batch_neg_poi_click_score,
                            self.network.neg_poi_category_ids:batch_neg_poi_category_ids,
                            self.network.neg_poi_category_mask: batch_neg_poi_category_mask,
                            self.network.query_type: batch_query_type,
                            self.network.query_city_id: batch_query_city_id,
                            self.network.query_geohash_id: batch_query_geohash_id,
                            self.network.query_week: batch_query_week,
                            self.network.query_day: batch_query_day,
                            self.network.bert_dropout_ratio: 0.1,
                            self.network.bert_output_dropout_ratio: 0.2
                        })
                    batch_poi_tokens = []
                    batch_poi_segments = []
                    batch_poi_masks = []
                    batch_pos_poi_tokens = []
                    batch_pos_poi_segments = []
                    batch_pos_poi_masks = []
                    batch_query_tokens = []
                    batch_query_segments = []
                    batch_query_masks = []
                    #new
                    batch_pos_poi_area_id = []
                    batch_pos_poi_geocode_id = []
                    batch_pos_poi_click_score = []
                    batch_pos_poi_category_ids = []
                    batch_pos_poi_category_mask = []
                    batch_neg_poi_area_id = []
                    batch_neg_poi_geocode_id = []
                    batch_neg_poi_click_score = []
                    batch_neg_poi_category_ids = []
                    batch_neg_poi_category_mask = []
                    batch_query_type = []
                    batch_query_city_id = []
                    batch_query_geohash_id = []
                    batch_query_week = []
                    batch_query_day = []
                batch_query_tokens.append(e.query["token_ids"])
                batch_query_segments.append(e.query["segment_ids"])
                batch_query_masks.append(e.query["mask_ids"])
                #pos_token_ids, pos_segment_ids, pos_mask_ids = self.get_poi_inputs(e.pos_tokens)
                pos_token_ids, pos_segment_ids, pos_mask_ids = self.get_poi_inputs(self.poi_token_ids[e.label][1])
                pos_area_id, pos_poi_geocode_id, pos_click_score, pos_poi_category_ids, pos_poi_category_mask = self.poi_token_ids[e.label][2:] #新增poi特征
                pos_poi_category_ids = pos_poi_category_ids.split(',')
                pospoi_category_mask = pos_poi_category_mask.split(',')
                #pos_token_ids, pos_segment_ids, pos_mask_ids = self.get_poi_inputs(self.poi_token_ids[0][1])
                batch_pos_poi_tokens.append(pos_token_ids)
                batch_pos_poi_segments.append(pos_segment_ids)
                batch_pos_poi_masks.append(pos_mask_ids)
                neg_token_ids, neg_segment_ids, neg_mask_ids = self.get_poi_inputs(self.poi_token_ids[e.poiid][1])
                neg_area_id, neg_poi_geocode_id, neg_click_score, neg_poi_category_ids, neg_poi_category_mask = self.poi_token_ids[e.poiid][2:]  #新增poi特征
                neg_poi_category_ids = neg_poi_category_ids.split(',')
                neg_poi_category_mask = neg_poi_category_mask.split(',')
                batch_poi_tokens.append(neg_token_ids)
                batch_poi_segments.append(neg_segment_ids)
                batch_poi_masks.append(neg_mask_ids)
                # new
                batch_pos_poi_area_id.append(pos_area_id)
                batch_pos_poi_geocode_id.append(pos_poi_geocode_id)
                batch_pos_poi_click_score.append(pos_click_score)
                batch_pos_poi_category_ids.append(pos_poi_category_ids)
                batch_pos_poi_category_mask.append(pospoi_category_mask)
                batch_neg_poi_area_id.append(neg_area_id)
                batch_neg_poi_geocode_id.append(neg_poi_geocode_id)
                batch_neg_poi_click_score.append(neg_click_score)
                batch_neg_poi_category_ids.append(neg_poi_category_ids)
                batch_neg_poi_category_mask.append(neg_poi_category_mask)
                batch_query_type.append(e.query["query_type"])
                batch_query_city_id.append(e.query["query_city_id"])
                batch_query_geohash_id.append(e.query["query_geohash_id"])
                batch_query_week.append(e.query["query_week"])
                batch_query_day.append(e.query["query_day"])
                step_num += 1
                if step_num % 500000 == 0:
                    tf.logging.info("run steps:{}, loss:{}".format( step_num,  loss))
                #print("learning", self.batch_dist_ratios.tolist()[0], poi_cos_dists.tolist()[0], batch_poi_dists[0])

    def step(self, state, action, reward, next_state, done, info, test_data, test_data_2):
        # collect train samples and training
        self.rb.add(state, action, reward, next_state, done, info)
        self.all_steps += 1
        self.t_step = (self.t_step + 1) % self.update_steps
        # 模型训练索引更新
        if self.t_step == 0 and len(self.rb) > self.batch_size:
            tf.logging.info("len of rb.memory:{}, {}".format(len(self.rb.memory), self.update_steps*self.batch_size*int(self.top_k/2)))
            experiences = self.rb.sample(self.update_steps*self.batch_size*int(self.top_k/2))
            self._learn(experiences)
            # self.restruct_all_index()
            self.restruct_all_index(is_test=False, poi_token_ids=self.poi_token_ids)
            #self.restruct_all_index(is_test=True, poi_token_ids=self.test_token_ids)
            topk_ratio = self.eval(test_data)
            tmp_topk_ratio = self.eval(test_data_2)
            if topk_ratio[2] > self.top5_ratio:
                self.top5_ratio = topk_ratio[2]
                print("top5_ratio:", self.top5_ratio)
                self.saver.save(self.sess, FLAGS.output_dir + "/new_save_model")

class Environment(object):
    def __init__(self, batch_size):
        # TODO: 如何传入？
        self.topk_labels = None

    def set_labels(self, topk_labels):
        self.topk_labels = topk_labels

    def _join_pois_info(self, action):
        all_poi_info = []
        # for poiid in action.poiids:
        #     poi_info = [poiid_info.get(i, "") for i in poiid]
        #     all_poi_info.append(poi_info)
        return all_poi_info

    def _cal_reward(self, topk_pois_info, action):
        # TODO:
        click_reward = self._cal_click_reward(action)
        # ndcg_reward = f(topk_pois_info)
        # reward = click_reward + ndcg_reward
        return click_reward

    def _cal_click_reward(self, action):
        click_num_ratio_of_pos_top10 = np.array([-0.43, -0.234, -0.144, -0.117, -0.057, -0.042, 0.032, 0.022, 0.015, 0.01])
        click_indice = np.argmax(self.topk_labels, axis=-1)
        click_indice[click_indice > 9] = 10
        click_rewards = click_num_ratio_of_pos_top10[click_indice] - click_num_ratio_of_pos_top10[3]
        return np.expand_dims(click_rewards, axis=1)  # [bs, 1]        

    def _cal_next_state(self, ):
        pass

    def step(self, action):
        self.topk_labels = action.labels
        topk_pois_info = self._join_pois_info(action)
        reward = self._cal_reward(topk_pois_info, action)
        #print(topk_pois_info,
        next_state = self._cal_next_state()
        done = False

        return next_state, reward, done, topk_pois_info

    def reset(self):
        # TODO: return a state (query)
        pass

class DataProcessor(object):
    def __init__(self):
        pass

    def convert_data(self, file_path):
        '''本地生成token 特征'''
        data = {"token_ids": [], "segment_ids": [], "mask_ids": [], "label": []}
        idx = 0
        all_token_ids = []
        all_segment_ids = []
        all_mask_ids = []
        all_labels = []
        with open(file_path, 'r') as f:
            for line in f:
                #if idx > 100:
                #    continue
                fields = line.strip('\r\n').split('\t')
                #g_searchid, g_country_code, cityid, poiid, query, pstnm, neg_list_str, g_birth_time, uniq_id = fields[1].strip().split('##')
                query, name, cityid, uniq_id = fields[:4]
                poiid_idx = poiid_idxs.get(uniq_id, 0)
                token_ids, segment_ids, mask_ids = convert_token_ids(query, FLAGS.max_seq_length, tokenizer)
                all_token_ids.append(token_ids)
                all_segment_ids.append(segment_ids)
                all_mask_ids.append(mask_ids)
                all_labels.append(poiid_idx)
                idx += 1
        data["token_ids"] = all_token_ids
        data["segment_ids"] = all_segment_ids
        data["mask_ids"] = all_mask_ids
        data["label"] = all_labels
        return data

    def load_data(self, file_path, batch_size):
        seq_length = FLAGS.max_seq_length
        name_to_features = {
            "token_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "mask_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "poi_token_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "city_id": tf.io.FixedLenFeature([1], tf.int64),
            "labels": tf.io.FixedLenFeature([1], tf.int64),
            "query_type": tf.io.FixedLenFeature([1], tf.int64), #new
            "query_city_id": tf.io.FixedLenFeature([1], tf.int64), 
            "query_geohash_id": tf.io.FixedLenFeature([1], tf.int64),
            "query_week": tf.io.FixedLenFeature([1], tf.int64), 
            "query_day": tf.io.FixedLenFeature([1], tf.int64),
        }
        def decorde(data):
            serial_exmp = tf.io.parse_single_example(data, name_to_features)
            label = serial_exmp.pop('labels')
            return (serial_exmp, label)
        input_files = tf.data.Dataset.list_files(file_path)
        input_data = input_files.apply(
            tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=2))
        # 数据解析
        parsed_data = input_data.map(map_func=decorde, num_parallel_calls=20)
        parsed_data = parsed_data.prefetch(batch_size*100).batch(batch_size)
        return parsed_data

def ppo(n_episodes=int(1e6), bs=128):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # TODO
    data_process = DataProcessor()
    train_data = data_process.load_data(FLAGS.train_tf_record_file, batch_size)
    test_data = data_process.load_data(FLAGS.test_tf_record_file, batch_size)
    test_data_2 = data_process.load_data(FLAGS.test_tf_record_file_2, batch_size)
    train_data = train_data.prefetch(buffer_size=50)
    test_data = test_data.prefetch(buffer_size=50)
    test_data_2 = test_data_2.prefetch(buffer_size=50)
    
    rewards = []                        # list containing rewards from each episode
    rewards_window = deque(maxlen=n_episodes)  # last 100 rewards
    max_t = 1
    epochs = FLAGS.num_train_epochs
    steps = 0
    print(FLAGS.test_tf_record_file)
    print(FLAGS.test_tf_record_file_2)
    #agent.restruct_all_index()
    topk_ratio = agent.eval(test_data)
    tmp_topk_ratio = agent.eval(test_data_2)
    if topk_ratio[2] > agent.top5_ratio:
        agent.top5_ratio = topk_ratio[2]
    for _ in range(epochs):
        is_epoch_end = True
        iterator = train_data.make_one_shot_iterator()
        data_iter = iterator.get_next() 
        while is_epoch_end:
            try:
                queries, labels = agent.convert_tensor_to_np(data_iter)
                if labels["labels"].shape[0] != batch_size:
                    is_epoch_end = False 
                    continue
            except tf.errors.OutOfRangeError:
                is_epoch_end = False
                continue
        
            state = BState(queries, None, None, labels) # [bs, q_emb_size]
            #print(state.labels)
            score = 0
            for t in range(max_t):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.step(state, action, reward, next_state, done, info, test_data, test_data_2)
                state = next_state
                score += reward

            rewards_window.append(score)       # save most recent score
            rewards.append(score)              # save most recent score
            
            steps += 1
    return rewards

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s - %(message)s')
for h in logger.handlers:
    h.setFormatter(formatter)
tf.logging.set_verbosity(tf.logging.INFO)
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

batch_size=FLAGS.search_batch_size
env = Environment(batch_size)
topk=30
# bert 中[SEP]对应的token id
sep_id = 102
# 默认城市，也是用于评测的城市
default_city = int(55000199)
# index = faiss.read_index("index.bin")
agent = Agent(buffer_size=FLAGS.replybuffer_size*batch_size*topk, top_k=topk,batch_size=batch_size, update_steps=FLAGS.update_steps)
rewards = ppo(n_episodes=10, bs=batch_size)
