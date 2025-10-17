#encoding=utf-8
import os
import sys
import argparse
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.regularizer import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer, Masking, Lambda, BatchNormalization
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from deep_model.gpu_embedding import GPUEmbedding
# from tensorflow.keras import mixed_precision
# import wandb
# from wandb.keras import WandbMetricsLogger
import json
import bert.tokenization as tokenization
import bert.modeling as modeling
import tensorflow_hub as hub
from tensorflow.keras.models import model
from autoint_layer import AutoIntLayer  
import tensorflow.compat.v1 as tf_v1


# wandb.login()

# tf.compat.v1.disable_eager_execution()


class DataProcessor(object):
    '''从tfrecord中解析训练/验证数据
    '''
    def __init__(self):
        self.feature_description = {
            "data_type": tf.io.FixedLenFeature([1], tf.string),
            "traceid_uid": tf.io.FixedLenFeature([1], tf.string),
            "group_len": tf.io.FixedLenFeature([1], tf.int64),
            "click_label": tf.io.FixedLenFeature([1], tf.int64),
            "order_label": tf.io.FixedLenFeature([1], tf.int64),
            "g_disp_query": tf.io.FixedLenFeature([32], tf.int64),
            "g_uid": tf.io.FixedLenFeature([1],tf.int64),
            "g_disp_area": tf.io.FixedLenFeature([1],tf.int64),
            "g_timestamp": tf.io.FixedLenFeature([1],tf.int64),
            "g_geohash": tf.io.FixedLenFeature([1],tf.string),
            "g_query_type": tf.io.FixedLenFeature([1],tf.int64),
            "component_ids": tf.io.FixedLenFeature([32], tf.int64),
            "pos_p_poi_id": tf.io.FixedLenFeature([1], tf.int64),
            "pos_p_geohash": tf.io.FixedLenFeature([1], tf.int64),
            "pos_p_category": tf.io.FixedLenFeature([10], tf.int64),
            "pos_p_name_address": tf.io.FixedLenFeature([32], tf.int64),
            "pos_onehot_pred_leaf": tf.io.FixedLenFeature([300], tf.int64),
            "pos_tree_score": tf.io.FixedLenFeature([1], tf.float32),
            "pos_p_layer": tf.io.FixedLenFeature([1], tf.int64),
            "pos_gBDTTop90FeatureList": tf.io.FixedLenFeature([90], tf.float32),
            "pos_gBDTRawFeatureList": tf.io.FixedLenFeature([34], tf.float32),
            "pos_gBDTDestPidCity": tf.io.FixedLenFeature([34], tf.float32),
            "pos_gBDTDestQueryCity": tf.io.FixedLenFeature([41], tf.float32),
            "pos_gBDTDestQueryGeo": tf.io.FixedLenFeature([34], tf.float32),
            "pos_gBDTDestPidQueryCity": tf.io.FixedLenFeature([17], tf.float32),
            "pos_gBDTStartQueryGeo": tf.io.FixedLenFeature([26], tf.float32),
            "pos_gBDTACDestQueryGeo": tf.io.FixedLenFeature([2], tf.float32),
            "pos_token_ids": tf.io.FixedLenFeature([32], tf.int64),
            "pos_segment_ids": tf.io.FixedLenFeature([32], tf.int64),
            "pos_mask_ids": tf.io.FixedLenFeature([32], tf.int64),
            "neg_p_poi_id": tf.io.FixedLenFeature([1], tf.int64),
            "neg_p_geohash": tf.io.FixedLenFeature([1], tf.int64),
            "neg_p_category": tf.io.FixedLenFeature([10], tf.int64),
            "neg_p_name_address": tf.io.FixedLenFeature([32], tf.int64),
            "neg_onehot_pred_leaf": tf.io.FixedLenFeature([300], tf.int64),
            "neg_tree_score": tf.io.FixedLenFeature([1], tf.float32),
            "neg_p_layer": tf.io.FixedLenFeature([1], tf.int64),
            "neg_gBDTTop90FeatureList": tf.io.FixedLenFeature([90], tf.float32),
            "neg_gBDTRawFeatureList": tf.io.FixedLenFeature([34], tf.float32),
            "neg_gBDTDestPidCity": tf.io.FixedLenFeature([34], tf.float32),
            "neg_gBDTDestQueryCity": tf.io.FixedLenFeature([41], tf.float32),
            "neg_gBDTDestQueryGeo": tf.io.FixedLenFeature([34], tf.float32),
            "neg_gBDTDestPidQueryCity": tf.io.FixedLenFeature([17], tf.float32),
            "neg_gBDTStartQueryGeo": tf.io.FixedLenFeature([26], tf.float32),
            "neg_gBDTACDestQueryGeo": tf.io.FixedLenFeature([2], tf.float32),
            "neg_token_ids": tf.io.FixedLenFeature([32], tf.int64),
            "neg_segment_ids": tf.io.FixedLenFeature([32], tf.int64),
            "neg_mask_ids": tf.io.FixedLenFeature([32], tf.int64),
        }
    def _parse_fn(self, data):
        serial_exmp = tf.io.parse_single_example(data, self.feature_description)
        click_label = serial_exmp['click_label']
        order_label = serial_exmp['order_label']
        return (serial_exmp,order_label)

    def input_fn_tfrecord(self, file_path):
        #匹配所有文件
        input_files = tf.data.Dataset.list_files(file_path)
        # 并行处理cycle_length个文件
        input_data = input_files.apply(
            tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=2))
        # 数据解析
        parsed_data = input_data.map(map_func=self._parse_fn, num_parallel_calls=20)
        return parsed_data


class FM(tf.keras.Model):
    '''FM部分
    '''
    def __init__(self):
        super(FM, self).__init__(name='fm_part')

        @tf.function
        def call(self,inputs):
            embed_inputs = inputs['fm_input']
            square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1,keepdims=True)) # (batch_size, 1, embed_dim)
            sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1,keepdims=True) # (batch_size, 1, embed_dim)
            cross_term = square_sum - sum_square
            second_order = 0.5 * tf.reduce_sum(cross_term, axis=-1, keepdims=False) # (batch_size, 1)
            second_order_emb = tf.squeeze(cross_term, [1])
            return second_order
        

class DNN(tf.keras.Model):
    '''DNN部分
    '''
    def __init__(self, hidden_units, activation_function='relu', dropout=0, name='deep_part'):
        super(DNN, self).__init__(name=name)
        self.hidden_units = hidden_units
        self.fc_layers = [Dense(units=unit, activation=activation_function) for unit in hidden_units]

    @tf.function
    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.hidden_units)):
            x = self.fc_layers[i](x)
        return x
    
class DeepFMModel(tf.keras.Model):
    def __init__(self, embedding_size, expret_dnn_hidden_units, tower_dnn_hidden_units, gate_dnn_hidden_units, 
                 num_experts,task_types, task_names, activation_function, l2_regualizer, dropout_rate, bert_config, bert_dropout_ratio, bert_output_dropout_ratio):
        super(DeepFMModel, self).__init__(name='deepfm_model')
        self.activation_function = activation_function
        self.l2_regualizer = l2_regualizer
        self.dropout_rate = dropout_rate
        self.expert_dnn_hidden_units = expret_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.num_experts = num_experts
        self.task_types = task_types
        self.task_names = task_names
        self.embedding_size = embedding_size
        self.start_ctr_loss = None
        self.start_ctcvr_loss = None
        self.end_ctr_loss = None
        self.end_ctcvr_loss = None
        self.loss = None
        #
        self.bert_layer = hub.KerasLayer("/data/bert_model/bert-en-uncased-l-6-h-256-a-4/2", trainable=True)
        
        #和词表一致
        self.embedding_dict = {
            "g_uid": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 16214818},
            "g_disp_area": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 4725},
            "g_timestamp": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 26},
            "q_geohash": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 103173},
            "g_disp_query": {"feat_len":32, "embedding_size": self.embedding_size, "input_dim": 40821},
            "component_ids": {"feat_len":10, "embedding_size": self.embedding_size, "input_dim": 16},
            "g_query_type": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 7},
            "p_geohash": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 103173},
            "p_name_address": {"feat_len":32, "embedding_size": self.embedding_size, "input_dim": 40821},
            "p_category": {"feat_len":10, "embedding_size": self.embedding_size, "input_dim": 470},
            "onehot_pred_leaf": {"feat_len":300, "embedding_size": self.embedding_size, "input_dim": 76501},
            "tree_score": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 2},
            #"p_layer": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 13},
            #"p_poi_id": {"feat_len":1, "embedding_size": self.embedding_size, "input_dim": 7158315},
        }

        self.pos_items = [
            "g_uid", "g_disp_area", "g_timestamp", "q_geohash", "pos_p_geohash",
            "component_ids", "g_query_type", 
            "pos_category" # "g_disp_query", "pos_p_name_address", "pos_onehot_pred_leaf"
        ]

        self.neg_items = [
            "g_uid", "g_disp_area", "g_timestamp", "q_geohash", "neg_p_geohash",
            "component_ids", "g_query_type", 
            "neg_category" # "g_disp_query", "neg_p_name_address", "neg_onehot_pred_leaf"
        ]

        # 映射关系
        self.kv_mapping = {
            "g_uid":"g_uid", 
            "g_disp_area":"g_disp_area",
            "g_timestamp":"g_timestamp", 
            "q_geohash":"q_geohash", 
            "g_disp_query": "g_disp_query",
            "component_ids": "component_ids",
            "g_query_type": "g_query_type",
            "pos_p_geohash": "p_geohash",
            "pos_p_name_address": "p_name_address",
            "pos_p_category": "p_category",
            "pos_onehot_pred_leaf": "onehot_pred_leaf",
            "pos_tree_score": "tree_score",
            "pos_token_ids": "p_name_address",
            "pos_p_layer": "p_layer",
            "neg_p_geohash": "p_geohash",
            "neg_p_name_address": "p_name_address",
            "neg_p_category": "p_category",
            "neg_onehot_pred_leaf": "onehot_pred_leaf",
            "neg_tree_score": "tree_score",
            "neg_token_ids": "p_name_address",
            "neg_p_layer": "p_layer",
        }

        self.sparse_items = [
            "g_uid", "g_disp_area", "g_timestamp", "q_geohash", "p_geohash",
            "g_disp_query","component_ids", "g_query_type", "p_name_address",
            "p_category", "onehot_pred_leaf"
        ]

        self.sparse_emb = {
            key: GPUEmbedding(
                input_dim=val["input_dim"],
                input_length=val["feat_len"],
                output_dim=val["embedding_size"],
                embedding_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.05),
                gpu_device=cuda_visible_devices,
                mask_zero=True,
                embedding_regualizer=l2(self.l2_regualizer),
            )for key, val in self.embedding_dict.items()
        }

        self.autoint_layer = AutoIntLayer(self.embedding_size) #初始化AutoInt层
        self.feat_weights = tf.v1.get_variable(
            "feat_weights", [90, self.embedding_size], 
            initializer=tf_v1.truncated_normal_initializer(stddev=0.5))
        
        self.expert_networks = [DNN(self.expert_dnn_hidden_units, self.activation_function, self.dropout_rate, name='expert_' + str(i)) for i  in range(self.num_experts)]
        self.gate_networks = [DNN(self.gate_dnn_hidden_units, self.activation_function, self.dropout_rate, name='gate_' + task_name) for task_name in task_names]
        self.gate_outs = [Dense(self.num_experts,use_bias=False, activation='softmax', name='gate_softmax_' + task_name) for task_name in task_names]
        self.tower_outputs = [Dnn(self.tower_dnn_hidden_units, self.activation_function, self.dropout_rate, name='tower_' + task_name) for task_name in task_names]
        self.logits = [Dense(1,use_bias=False,  name='logits_' + task_name) for task_name in task_names]

    @tf.function
    def call(self, inputs, training=False):

        num_tasks = len(self.task_names)
        if num_tasks <=1:
            raise ValueError('task_names must grater than 1')
        
        if self.num_experts <= 1:
            raise ValueError('num_experts must grater than 1')
        
        if len(self.task_types != num_tasks):
            raise ValueError('task_types must be the same length as task_names')
        
        for task_type in self.task_types:
            if task_type not in ['binary','regression']:
                raise ValueError('task_type must be binary or regression, {} is illegal'.format(task_type))

        # DNN输入
        # 输入特征
        # pos_tree_score = inputs['pos_tree_score']
        # neg_tree_score = inputs['neg_tree_score']
        pos_gbdt_top_feat = inputs["pos_gBDTTop90FeatureList"]
        neg_gbdt_top_feat = inputs["neg_gBDTTop90FeatureList"]

        # pos_text_embedding = self.bertNet(inputs['pos_token_ids'], inputs['pos_mask_ids'], inputs['pos_segment_ids'])
        # neg_...
        pos_text_inputs = {'input_mask':tf.cast(inputs['pos_mask_ids'], dtype=tf.int32),
                            'input_type_ids': tf.cast(inputs['pos_segment_ids'], dtype=tf.int32),
                            'input_word_ids': tf.cast(inputs['pos_token_ids'], dtype=tf.int32)
                            }
        neg_text_inputs = {'input_mask':tf.cast(inputs['neg_mask_ids'], dtype=tf.int32),
                            'input_type_ids': tf.cast(inputs['neg_segment_ids'], dtype=tf.int32),
                            'input_word_ids': tf.cast(inputs['neg_token_ids'], dtype=tf.int32)
                            }
        pos_text_output = self.bert_layer(pos_text_inputs)
        neg_text_output = self.bert_layer(neg_text_inputs)

        pos_fm_emb = tf.concat([self.sparse_emb[self.kv_mapping[k]](inputs[k]) for k in self.pos_items], axis=1) # N * M * d
        pos_feat_emb = self.cal_num_feature_embedding(pos_gbdt_top_feat, self.feat_weights) # N * 90 * d
        pos_concat_emb = tf.concat([pos_fm_emb, pos_feat_emb], axis=1)
        pos_autoint_emb = self.autoint_layer(pos_concat_emb, training)
        pos_autoint_emb2 = self.autoint_layer(pos_autoint_emb, training)
        pos_autoint_emb = tf.reshape(pos_autoint_emb2,[-1, pos_autoint_emb2.shape[1] * pos_autoint_emb2.shape[2]])
        pos_dnn_input = tf.concat([pos_autoint_emb, pos_text_output["pooled_output"]], axis=-1)

        neg_fm_emb = tf.concat([self.sparse_emb[self.kv_mapping[k]](inputs[k]) for k in self.neg_items], axis=1) # N * M * d
        neg_feat_emb = self.cal_num_feature_embedding(neg_gbdt_top_feat, self.feat_weights) # N * 90 * d
        neg_concat_emb = tf.concat([neg_fm_emb, neg_feat_emb], axis=1)
        neg_autoint_emb = self.autoint_layer(neg_concat_emb, training)
        neg_autoint_emb2 = self.autoint_layer(neg_autoint_emb, training)
        neg_autoint_emb = tf.reshape(neg_autoint_emb2,[-1, neg_autoint_emb2.shape[1] * neg_autoint_emb2.shape[2]])
        neg_dnn_input = tf.concat([neg_autoint_emb, neg_text_output["pooled_output"]], axis=-1)

        outputs = []
        for dnn_input in (pos_dnn_input, neg_dnn_input):
            # build expert layer
            expert_outs = []
            for i in range(self.num_experts):
                expert_outs.append(self.expert_networks[i](dnn_input, training))
            expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outs)
            # print(expert.concat.shape)

            task_outs = []
            for i in range(num_tasks):
                gate_out = self.gate_outs[i](self.gate_networks[i](dnn_input, training))
                #build gate layer
                gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)
                # gate multiply the expert
                gate_mul_expert = Lambda(lambda x: tf.reduce_sum(x[0]* x[1],axis=1,keepdims=False), name='gate_mul_expert_'+ self.task_names[i])([expert_concat,gate_out])
                # print(gate_mul_expert.shape)
                # bulid tower
                logit = self.logits[i](self.tower_outputs[i](gate_mul_expert,training))
                logit = tf.reshape(logit,[-1, 1])
                task_outs.append(logit)

            # 输出
            start_ctr_logits, start_cvr_logits, end_ctr_logits, end_cvr_logits = task_outs
            # start_ctcvr_logits = tf.multiply(start_ctr_logits, start_cvr_logits)
            # end_ctcvr_logits = tf.multipyl(end_ctr_logits, end_cvr_logits)

            outputs.extend([start_ctr_logits, start_cvr_logits, end_ctr_logits, end_cvr_logits])

        #对应self.feature_description
        qtype = inputs['g_query_type']
        data_type = inputs['data_type']

        c_tensor = tf.fill(tf.shape(data_type), 'click')
        click_mask = tf.cast(tf.equal(data_type, c_tensor), dtype=tf.float32)
        s_tensor = tf.fill(tf.shape(qtype), tf.constant(4, dtype=tf.int64))
        qtype_mask = tf.cast(tf.equal(qtype, s_tensor), dtype= tf.float32)

        outputs.extend([click_mask, qtype_mask])
        prob = tf.stack(outputs)

        return prob
    
    def cal_num_feature_embedding(self, gbdt_top_feat, embedding_table):
        batch_size = gbdt_top_feat.shape[0]
        a = tf.range(start=0, limit=90) #为90维实值特征，生成0-89的索引，查embedding_table用
        b = tf.expand_dims(a, axis=0)
        feature_idx = tf.tile(b, multiples=[batch_size, 1]) # N* 90
        feature_emb = tf.nn.embedding_lookup(embedding_table, feature_idx) # N*90*d
        gbdt_top_feat = tf.expand_dims(gbdt_top_feat, axis=-1) # N * 90 * 1
        output = tf.multiply(feature_emb, gbdt_top_feat) # N * 90 * d

        return output
    
    def multi_task_loss(self, y_true, y_pred):
        pos_start_ctr_logits, pos_start_cvr_logits, pos_end_ctr_logits, pos_end_cvr_logits, neg_start_ctr_logits, neg_start_cvr_logits, neg_end_ctr_logits, neg_end_cvr_logits, click_mask, qtype_mask = tf.unstack(y_pred)
        #起点
        self.start_ctr_loss = click_loss_weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_start_ctr_logits), logits=pos_start_ctr_logits-neg_start_ctr_logits) * qtype_mask *click_mask)
        self.start_ctcvr_loss = order_loss_weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_start_cvr_logits), logits=pos_start_cvr_logits-neg_start_cvr_logits) * qtype_mask * tf.cast(tf.math.logical_not(tf.cast(click_mask,tf.bool)),tf.float32))
        #终点
        self.end_ctr_loss = click_loss_weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_end_ctr_logits), logits=pos_end_ctr_logits-neg_end_ctr_logits) * tf.cast(tf.math.logical_not(tf.cast(qtype_mask,tf.bool)),tf.float32) * click_mask)
        self.end_ctcvr_loss = order_loss_weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_end_cvr_logits), logits=pos_end_cvr_logits-neg_end_cvr_logits) * tf.cast(tf.math.logical_not(tf.cast(qtype_mask,tf.bool)),tf.float32) *  tf.cast(tf.math.logical_not(tf.cast(click_mask,tf.bool)),tf.float32))
        self.loss = self.start_ctr_loss + self.start_ctcvr_loss + self.end_ctr_loss + self.end_ctcvr_loss
        return self.loss
    
class CustomSchedule(tf.keras.optimizer.schdules.LearningRateSchedule):
    #带warmup的学习率衰减，https://arxiv.org/pdf/1706.03762.pdf
    def __init__(self, d_model=16, warmup_steps=36000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.lr = None
        self.cur_epoch = 0
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        self.lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) * tf.math.pow(0.1, self.cur_epoch * 2 + 1) #第一个epoch，乘以0.1， 第二个epoch，乘以0.001
        return self.lr

    def setEpoch(self, epoch):
        self.cur_epoch = epoch

def train():
    ## 数据处理 ##
    # 加载数据
    data_processor = DataProcessor()
    train_data = data_processor.input_fn_tfrecord(train_path)
    valid_data = data_processor.input_fn_tfrecord(test_path)


    # 设置数据读取线程，根据CPU核数确定
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 20
    train_data = train_data.with_options(options)
    valid_data = valid_data.with_options(options)

    # 数据batch化
    train_data = train_data.prefetch(100 * batch_size).shuffle(10000).batch(batch_size)
    valid_data = valid_data.prefetch(50 * batch_size).batch(batch_size)

    ## 定义模型 ##
    # 模型实例化
    model = DeepFMModel(
        embedding_size=embedding_size,
        expret_dnn_hidden_units=expert_dnn_hidden_units,
        tower_dnn_hidden_units=tower_dnn_hidden_units,
        gate_dnn_hidden_units=gate_dnn_hidden_units,
        num_experts=num_experts,
        task_types=task_types,
        task_names=task_names,
        activation_function=activation_function,
        l2_regualizer=l2_regualizer,    
        dropout_rate=dropout_rate,
        bert_config=bert_config,
        bert_dropout_ratio=0.1,
        bert_output_dropout_ratio=0.2
    )

    # 学习率
    learning_rate = CustomSchedule()
    # learning_rate = 8e-5

    # 模型编译
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(
        optimizer=model.optimizer, 
        loss=model.multi_task_loss,
        run_eagerly=False
    )

    class PrintLR(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            print("\nLearning rate for epoch {} step {} is {}:".format( learning_rate.cur_epoch, batch, learning_rate.lr))

    class PrintTowerLoss(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            print("\nstep {}, start_ctr_loss: {}, start_cvr_loss: {}, end_ctr_loss: {}, end_cvr_loss: {}, loss: {}".format( 
                batch, model.start_ctr_loss, model.start_ctcvr_loss, model.end_ctr_loss, model.end_ctcvr_loss, model.loss))

    class SetLearningRate(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            learning_rate.setEpoch(epoch)
            print('\nLearning rate for CustomSchedule is set to {}'.format(epoch))

    # checkpoint路径
    ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')

    # 定义callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor='val_loss',  mode='min', save_best_only=True, save_weights_only=True, verbose=1),
        PrintLR(),
        PrintTowerLoss(),
        SetLearningRate()
    ]
    # WandbMetricsLogger(log_freq=10)

    ## 断点续训 ##
    if not from_scratch and os.path.exists(ckpt_path + '.index'):
        print('load model finished.')
        model.load_weights(ckpt_path)


    # configure = tf.ConfigProto()
    # configure.gpu_options.allow_growth = True
    # configure.gpu_options.per_process_gpu_memory_fraction = 1
    # configure.log_device_placement = False
    # sess = tf.Session(config=configure)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # 获取变量集，计算梯度
    #trainable_vars = model.trainable_variables
    # 记录训练前权重
    #grads2 = []
    #for var in  trainable_vars:
    #   grads2.append(tf.cast(var.numpy()*0., dtype=tf.float32))
    #       #按梯度更新所有权重变量
    #model.optimizer.apply_gradients(zip(grads2, trainable_vars))

    ## 模型训练 ##
    model.fit(train_data, epochs=epoch, validation_data=valid_data, callbacks=callbacks)

    # wandb.log({
    # 'epoch': epoch, 
    # 'traing_loss': model.loss, 
    # 'start_ctr_loss': model.start_ctr_loss, 
    # 'start_cvr_loss': model.start_ctcvr_loss, 
    # 'end_ctr_loss': model.end_ctr_loss, 
    # 'end_cvr_loss': model.end_ctcvr_loss
    # })

    return

def eval():
    # 模型加载
    model = DeepFMModel(
        embedding_size=embedding_size,
        expret_dnn_hidden_units=expert_dnn_hidden_units,
        tower_dnn_hidden_units=tower_dnn_hidden_units,
        gate_dnn_hidden_units=gate_dnn_hidden_units,
        num_experts=num_experts,
        task_types=task_types,
        task_names=task_names,
        activation_function=activation_function,
        l2_regualizer=l2_regualizer,    
        dropout_rate=dropout_rate,
        bert_config=bert_config,
        bert_dropout_ratio=0.0,
        bert_output_dropout_ratio=0.0
    )
    ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')
    model.load_weights(ckpt_path)

    # 数据加载
    data_processor = DataProcessor()
    test_data = data_processor.input_fn_tfrecord(test_path)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 20
    test_data = test_data.with_options(options)
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(1)

    # 预测、落盘
    start_write_buf = []
    end_write_buf = []
    f_start = open(pred_path + '.start', 'w')
    f_end = open(pred_path + '.end', 'w')

    # 一次预测一个batch
    for data, _ in test_data:
        qtype = data["g_query_type"].numpy().tolist()
        data_type = data["data_type"].numpy().tolist()
        groupId = data["traceid_uid"].numpy().tolist()
        click_label = data["click_label"].numpy().tolist()
        order_label = data["order_label"].numpy().tolist()
        # 输出维度是8:
        pred = model.predict(data, batch_size=batch_size)
        start_pos_ctr_logits = pred[0].flatten()  # 为啥要flatten？
        start_pos_ctcvr_logits = pred[1].flatten()
        end_pos_ctr_logits = pred[2].flatten()
        end_pos_ctcvr_logits = pred[3].flatten()

        #start_neg_ctr_logits = pred[4].flatten()  
        #start_neg_ctcvr_logits = pred[5].flatten()
        #end_neg_ctr_logits = pred[6].flatten()
        #end_neg_ctcvr_logits = pred[7].flatten()

        for q,d,g, start_order_label, start_order_pred, start_click_label, start_click_pred, end_order_label, end_order_pred, end_click_label, end_click_pred in \
            zip(qtype, data_type, groupId,order_label, start_pos_ctcvr_logits, click_label, start_pos_ctr_logits, order_label, end_pos_ctcvr_logits, click_label, end_pos_ctr_logits):
            # # bytes转str
            if q[0] == 4:
                start_write_buf.append(' '.join([str(g[0]), str(start_order_label[0]), str(start_order_pred), str(start_click_label[0]), str(start_click_pred),
                                    str(end_order_label[0]), str(end_order_pred), str(end_click_label[0], str(end_click_pred))]) + '\n')
                if len(start_write_buf) % 50000 == 0:
                    f_start.writelines(start_write_buf)
                    start_write_buf = []
            else:
                end_write_buf.append(' '.join([str(g[0]), str(start_order_label[0]), str(start_order_pred), str(start_click_label[0]), str(start_click_pred),
                                    str(end_order_label[0]), str(end_order_pred), str(end_click_label[0], str(end_click_pred))]) + '\n')
                if len(end_write_buf) % 50000 == 0:
                    f_end.writelines(end_write_buf)
                    end_write_buf = []

    if len(start_write_buf) > 0:
        f_start.writelines(start_write_buf)
    f_start.close()

    if len(end_write_buf) > 0:
        f_end.writelines(end_write_buf)
    f_end.close()

    return 

def save():
    # 模型加载
    model = DeepFMModel(
        embedding_size=embedding_size,
        expret_dnn_hidden_units=expert_dnn_hidden_units,
        tower_dnn_hidden_units=tower_dnn_hidden_units,
        gate_dnn_hidden_units=gate_dnn_hidden_units,
        num_experts=num_experts,
        task_types=task_types,
        task_names=task_names,
        activation_function=activation_function,
        l2_regualizer=l2_regualizer,    
        dropout_rate=dropout_rate,
        bert_config=bert_config,
        bert_dropout_ratio=0.0,
        bert_output_dropout_ratio=0.0
    )
    ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')
    model.load_weights(ckpt_path)

    # 自动built输入
    data_processor = DataProcessor()
    test_data = data_processor.input_fn_tfrecord(test_path)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 20
    test_data = test_data.with_options(options)
    test_data = test_data.batch(batch_size)
    test_data = test_data.prefetch(1)
    for data, _ in test_data:
        _ = model.predict(data, batch_size=batch_size)
        break

    ## 保存模型 ##
    model.save(model_path, save_format="tf")

    return 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='cuda visible devices')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--from_scratch', action='store_true', default=False, help='Train from scratch or not')
    parser.add_argument('--lr', type=float, default=-1.0, help='initial learning rate')
    parser.add_argument('--expert_dnn_hidden_units', type=int, default=[512,256,128,64], nargs='+', help='expert DNN hidden units')
    parser.add_argument('--tower_dnn_hidden_units', type=int, default=[32,16], nargs='+', help='tower DNN hidden units')
    parser.add_argument('--gate_dnn_hidden_units', type=int, default=[32,16], nargs='+', help='gate DNN hidden units')
    parser.add_argument('--num_experts', type=int, default=3, help='number of experts')
    parser.add_argument('--task_types', type=str, default=['binary','binary','binary','binary'], nargs='+', help='task types')
    parser.add_argument('--task_names', type=str, default=['start_ctr','start_ctcvr','end_ctr','end_ctcvr'], nargs='+', help='task names')
    parser.add_argument('--act', type=str, default='elu', help='activation function')
    parser.add_argument('--l2', type=float, default=1e-6, help='l2 regularizer')
    parser.add_argument('--dr', type=float, default=-1.0, help='dropout rate')
    parser.add_argument('--bs', type=int, default=2048, help='batch size')
    parser.add_argument('--ep', type=int, default=1, help='epoch')
    parser.add_argument('--es', type=int, default=16, help='embedding size')
    parser.add_argument('--click_loss_weight', type=float, default=1.0, help='click loss weight')
    parser.add_argument('--order_loss_weight', type=float, default=1.0, help='order loss weight')
    parser.add_argument('--train_path', type=str, default='', help='train dataset data path')
    parser.add_argument('--test_path', type=str, default='', help='test dataset data path')
    parser.add_argument('--ckpt_dir', type=str, default='../model_mtr_pairwise_ple_mmoe_1/checkpoints', help='checkpoint dir')
    parser.add_argument('--log_dir', type=str, default='../model_mtr_pairwise_ple_mmoe_1/logs', help='log saved path')
    parser.add_argument('--model', type=str, default='../model_mtr_pairwise_ple_mmoe_1/1', help='model saved path')
    parser.add_argument('--pred', type=str, default='../model_mtr_pairwise_ple_mmoe_1/mtr.pred.res', help='predict result saved path')
    parser.add_argument('--start_or_end', type=int, default=1, help='eval data, 2=start or 1=end')
    parser.add_argument('--bert_config', type=str, default='./bert_config/config.json', help='bert config file path')
    parser.add_argument('--bert_vocab', type=str, default='./bert_config/vocab.txt', help='bert vocab file path')
    args = parser.parse_args()

    cuda_visible_devices = args.gpus
    train_or_eval = args.mode
    from_scratch = args.from_scratch
    learning_rate = args.lr
    expert_dnn_hidden_units = args.expert_dnn_hidden_units
    tower_dnn_hidden_units = args.tower_dnn_hidden_units
    gate_dnn_hidden_units = args.gate_dnn_hidden_units
    num_experts = args.num_experts
    task_types = args.task_types
    task_names = args.task_names
    activation_function = args.act
    l2_regualizer = args.l2
    dropout_rate = args.dr
    batch_size = args.bs
    epoch = args.ep
    embedding_size = args.es
    click_loss_weight = args.click_loss_weight
    order_loss_weight = args.order_loss_weight
    train_path = args.train_path
    test_path = args.test_path
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    model_path = args.model
    pred_path = args.pred
    start_or_end = args.start_or_end
    bert_config = args.bert_config
    # wandb.init(project="mmoe_mtr_pairwise_ple_mmoe_1")


    # 列出设备上的GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    # 设备使用的gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    tf.config.experimental_run_functions_eagerly(True)

    # 按需申请显存空间
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if train_or_eval == 'train':
        train()
    elif train_or_eval == 'eval':
        eval()
    elif train_or_eval == 'save':
        save()
    else:
        raise ValueError('train_or_eval must in (train, eval, save)')











