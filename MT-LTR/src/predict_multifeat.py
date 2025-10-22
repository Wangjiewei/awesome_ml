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
from collections import deque, namedtuple, defaultdict, OrderedDict
import tensorflow.compat.v1 as tf
import tensorflow as tf_2
import mkl
import time
import os
import gc
import random
import faiss
from faiss import METRIC_INNER_PRODUCT
from sklearn.preprocessing import normalize
#distances = cosine_distances(eq, epoi)
from sklearn.metrics.pairwise import cosine_distances

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tokenization 
import modeling_rl as modeling
from modeling_rl import create_initializer
mkl.get_max_threads()
tf.disable_eager_execution() # 执行错误就restart jupyter kernel
#tf.config.experimental_run_functions_eagerly(True)
#json.encoder.FLOAT_REPR = lambda x: format(x, '.8f')

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
## processing data parameters
flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")
flags.DEFINE_integer(
    "max_seq_length", 72,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_string("train_tf_record_file", 'train.tf_record', "The train file that the BERT model was trained on.")
flags.DEFINE_string("dev_tf_record_file", 'dev.tf_record', "The evaluation file that the BERT model was evaluated on.")
flags.DEFINE_string("test_tf_record_file", 'test.tf_record', "The test file that the BERT model was tested on.")
flags.DEFINE_string("log_name", "tf_logs.txt", "log file name")
flags.DEFINE_string("predict_result_name", "twbrt_results.tsv", "predict result file name")
flags.DEFINE_string("predict_vector_name", "twbrt_vectors.tsv", "predict vector file name")
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
flags.DEFINE_bool("use_fp16", False, "Whether to use fp16.")
flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 128, "Total batch size for predict.")
flags.DEFINE_integer("index_size", 128, "index hidden size.")
flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "How often to save the model checkpoint.")
flags.DEFINE_integer("keep_checkpoint_max", 3, "How many checkpoint file will be saved")

# TPU parameters: not be used in this experiment
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

gpu_ids = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

def convert_token_ids(example, max_seq_len, tokenizer):
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

def train_index(all_poi_embs, index_type='flat'):
    d = 128
    epoi = normalize(all_poi_embs, axis=1, norm='l2')
    if index_type == 'hnsw':
        # hnsw
        M = 10  # the number of neighbors used in the graph.
        index = faiss.IndexHNSWFlat(d, M, METRIC_INNER_PRODUCT)  # , efConstruction=10, efSearch=4)
        index = faiss.index_cpu_to_all_gpus(index)
    else:
        # flat with cosine (Exact Search)
        index = faiss.IndexFlatIP(d)
        index = faiss.index_cpu_to_all_gpus(index)

    index.train(epoi)
    index.add(epoi)
    return index

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

        self.query_pooling_token_tensor_text,_,_,_,_,_,_ = self.get_bert_out(self.query_input_ids, self.query_input_mask, self.query_segment_ids)
        self.pos_pooling_token_tensor_text, self.bert_seq_layer, self.bert_attention_score, self.bert_output, self.input_ids_input, self.input_mask_input,self.segment_ids_input = self.get_bert_out(self.pos_poi_input_ids, self.pos_poi_input_mask, self.pos_poi_segment_ids)
        self.neg_pooling_token_tensor_text,_,_,_,_,_,_ = self.get_bert_out(self.neg_poi_input_ids, self.neg_poi_input_mask, self.neg_poi_segment_ids)

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

        return bert_pooling_token_tensor, bert_seq_layer, bert_attention_score, bert_output_layer, input_ids, input_mask,segment_ids

class PoiHNSW(object):
    def __init__(self, index_dir="fasis_indexs"):
        self.index_dir = index_dir

    def _read_vec(self, line):
        line = line.strip().split("\t")
        poiid, city, token_ids,area_id, poi_geocode_id, click_score, poi_category_ids, poi_category_mask = line
        #return poiid, city, [int(i) for i in token_ids.split(";")]
        return int(poiid), int(city), token_ids, int(area_id), int(poi_geocode_id), float(click_score), poi_category_ids, poi_category_mask

    def product_all_index(self, poi_vec_path):
        all_vecs = []
        all_infos = {}
        poiid_index = {}
        all_token_ids = []
        with open(poi_vec_path, 'r') as f:
            idx = 0
            for line in f:
                #city, info, poiid, vecs, token_ids = self._read_vec(line)
                poiid, city, token_ids, area_id, poi_geocode_id, click_score, poi_category_ids, poi_category_mask = self._read_vec(line)
                if not city:
                    continue
                poiid_index[idx] = poiid
                all_infos[poiid] = city
                all_token_ids.append((city, token_ids, area_id, poi_geocode_id, click_score, poi_category_ids, poi_category_mask))
                idx += 1
                #if idx > 10000:
                #    break
            print("all_poi_nums:", idx)
        return all_infos, poiid_index, all_token_ids
    def load_vectors(self, poi_vec_path):
        poiid_idx = {}
        idx_poiid = {}
        all_poi_vectors = []
        with open(poi_vec_path, "r") as f:
            idx = 0
            for line in f:
                line = line.strip()
                data = json.load(line)
                poiid = data["poi_id"]
                cityid = data["city_id"]
                vector = data["vector"]
                all_poi_vectors.append(vector)
                poiid_idx[poiid] = idx
                idx_poiid[idx] = poiid
        return all_poi_vectors, poiid_idx, idx_poiid 

class Agent(object):
    def __init__(self, buffer_size, batch_size, update_steps, top_k=10, reconstruction_index_thres=1.5):
        self.optimizer = tf.keras.optimizers.Adam()
        self.index = None#train_index(poi_vecs)
        self.batch_size = batch_size

        #init = tf.global_variables_initializer()
        self.network = EBRNetworkNew(bert_config)
        self.saver = tf.train.Saver()
        self.top_k = top_k
        self.sess = self._init_session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #self.network = network
        #self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.init_checkpoint))
        if(FLAGS.init_checkpoint is not None):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.init_checkpoint))
            #tvars = tf.trainable_variables()
            #initialized_variable_names = {}
            #(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
            #tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

    def _init_session(self):
        configure = tf.ConfigProto()
        configure.gpu_options.allow_growth=True
        configure.gpu_options.per_process_gpu_memory_fraction = 1
        configure.log_device_placement = False
        return tf.Session(config=configure)

    def _create_token_pooling_embedding(self, pois):
        poi_embs = self.sess.run([self.network.pos_pooling_token_tensor], feed_dict={
                 self.network.pos_poi_input_ids: pois["token_ids"],
                 self.network.pos_poi_input_mask: pois["mask_ids"],
                 self.network.pos_poi_segment_ids: pois["segment_ids"],
                self.network.pos_poi_geocode_id: pois["geocode_id"],
                self.network.pos_poi_click_score: pois["click_score"],
                self.network.pos_poi_category_ids: pois["category_ids"],
                self.network.pos_poi_category_mask: pois["category_mask"],
                self.network.bert_dropout_ratio: 0.0,
                self.network.bert_output_dropout_ratio: 0.0
             })
        return poi_embs[0]

    def restruct_all_index(self):
        all_embeds = []
        info_size = 100
        batch_token_ids = []
        batch_segment_ids = []
        batch_mask_ids = []
        batch_poiids = []
        batch_poi_geocode_id = []
        batch_click_score = []
        batch_poi_category_ids = []
        batch_poi_category_mask = []
        start_time = int(time.time())
        f = open(os.path.join(FLAGS.output_dir, 'poi_vec_info_all') , "w")
        print("start_reindex")
        for idx, input_features in enumerate(poi_token_ids):
            new_token_ids, segment_ids, mask_ids = self.get_poi_inputs(input_features[1])
            #print(idx, new_token_ids, segment_ids, mask_ids)
            batch_token_ids.append(new_token_ids)
            batch_segment_ids.append(segment_ids)
            batch_mask_ids.append(mask_ids)
            batch_poiids.append(idx) 
            batch_poi_geocode_id.append(input_features[3])
            batch_click_score.append(input_features[4])
            batch_poi_category_ids.append(input_features[5].split(','))
            batch_poi_category_mask.append(input_features[6].split(','))
            if idx > 0 and idx % info_size == 0:
                poi_vecs = self._create_token_pooling_embedding({"token_ids": batch_token_ids, "segment_ids": batch_segment_ids, "mask_ids": batch_mask_ids,  "geocode_id": batch_poi_geocode_id, "click_score": batch_click_score, "category_ids": batch_poi_category_ids, "category_mask": batch_poi_category_mask})
                poi_vecs = poi_vecs.astype(np.float32).tolist()
                for id_index, poi_vec in zip(batch_poiids, poi_vecs):
                    poi_vec = [round(i, 8) for i in poi_vec]
                    hnsw_data = {"poi_id": poiid_index[id_index], "city_id": poi_city[poiid_index[id_index]], "vector": poi_vec}
                    f.write("%s\n" % json.dumps(hnsw_data))
                batch_mask_ids = []
                batch_segment_ids = []
                batch_token_ids = []
                batch_poiids = []
                batch_poi_geocode_id = []
                batch_click_score = []
                batch_poi_category_ids = []
                batch_poi_category_mask = []
        if len(batch_token_ids) > 0:
            poi_vecs = self._create_token_pooling_embedding({"token_ids": batch_token_ids, "segment_ids": batch_segment_ids, "mask_ids": batch_mask_ids,  "geocode_id": batch_poi_geocode_id,     "click_score": batch_click_score, "category_ids": batch_poi_category_ids, "category_mask": batch_poi_category_mask})
            poi_vecs = poi_vecs.astype(np.float32).tolist()
            #print("debug_poi_info", ";".join(map(str,batch_token_ids[0])))
            #print("debug_poi_segm", ";".join(map(str,batch_segment_ids[0])))
            #print("debug_poi_mask", ";".join(map(str,batch_mask_ids[0])))
            #print("debug_poi_geohash", batch_poi_geocode_id[0])
            #print("debug_poi_click", batch_click_score[0])
            #print("debug_poi_cateid", batch_poi_category_ids[0])
            #print("debug_poi_catemsk", batch_poi_category_mask[0])
            #print("debug_poi_embeds", ",".join(map(str,poi_vecs[0])))
            #print("debug_poi_embed1", ",".join(map(str,poi_embs_text[0][0])))
            #print("debug_poi_embed2", ",".join(map(str,clickscore_embs[0][0])))
            #print("debug_poi_embed3", ",".join(map(str,cate_embs[0][0])))
            #print("debug_poi_embed4", ",".join(map(str,geohash_embs[0][0])))
            #print("debug_poi_embed5", ",".join(map(str,bert_seq_out[0])))
            #print("debug_poi_embed6", ",".join(map(str,bert_attention[0])))
            #print("debug_poi_embed7", ",".join(map(str,bert_output[0][0])))
            #print("debug_poi_embed8", ";".join(map(str,token_ids_input[0])))
            #print("debug_poi_embed9", ";".join(map(str,token_mask_input[0])))
            #print("debug_poi_embed10", ";".join(map(str,token_seg_input[0])))
            for id_index, poi_vec in zip(batch_poiids, poi_vecs):
                poi_vec = [round(i, 8) for i in poi_vec]
                hnsw_data = {"poi_id": poiid_index[id_index], "city_id": poi_city[poiid_index[id_index]], "vector": poi_vec}
                f.write("%s\n" % json.dumps(hnsw_data))
            #all_embeds.extend(poi_vecs.tolist())
        #print(all_embeds)

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

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
# poi_info_path = os.path.join(FLAGS.data_dir, FLAGS.info_file)
# poi_infos = PoiInfo(poi_info_path)
start_time=int(time.time())
print('start_time: ', start_time)
poihnsw = PoiHNSW()
all_vecs = []
poi_city, poiid_index, poi_token_ids = poihnsw.product_all_index(FLAGS.vector_file)
print(len(poi_token_ids))
print("hnsw start time:", int(time.time()) - start_time)
batch_size=128
#sep_id=102
sep_id=5
# index = faiss.read_index("index.bin")
agent = Agent(buffer_size=int(2e3)*batch_size*10, batch_size=batch_size, update_steps=int(2e3))
agent.restruct_all_index()
