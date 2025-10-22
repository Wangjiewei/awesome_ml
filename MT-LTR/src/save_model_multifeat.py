# coding=utf-8
from numpy.core.fromnumeric import shape
import tensorflow as tf
import os
import numpy as np
import time
import sys
import json
#sys.path.append("..")
import tokenization
import modeling
from modeling import create_initializer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

is_training=False
use_one_hot_embeddings=False
#batch_size=16
batch_size=400

#cty_code = "BR"
cty_code = "MX"

vocab_file="../pretrain_lm/electra-base-portuguese-uncased-brwac/vocab4.txt"
bert_config_file="../pretrain_lm/electra-base-portuguese-uncased-brwac/config.json4"
if(cty_code == "MX"):
    vocab_file="../pretrain_lm/spanish_uncased_L-3_H-768_A-12/vocab4.txt"
    bert_config_file="../pretrain_lm/spanish_uncased_L-3_H-768_A-12/config.json2"


model_ckpt = "../models/siamese_br_new_finetune_safe_predict"
model_export_dir="../serving_model/siamese_br_v5_safe/1"
if(cty_code == "MX"):
    model_ckpt = "../models/siamese_mx_new_finetune_safe_predict"
    model_export_dir="../serving_model/siamese_mx_v5_safe/1"

et_hidden_size = 128
et_query_type_class_num = 2
et_geohash_class_num = 540899
et_week_bucket_class_num = 8
et_day_bucket_class_num = 7
if(cty_code == "MX"):
    et_geohash_class_num = 172830


max_seq_length=32
#output_dir="./retrieval_model"
do_lower_case=True
#final_output_file="./query_vector_0118"

# global graph
input_ids_q, input_mask_q, label_ids_q, segment_ids_q = None, None, None, None
#print(output_dir)
#print('checkpoint path:{}'.format(os.path.join(output_dir, "checkpoint")))
#if not os.path.exists(os.path.join(output_dir, "checkpoint")):
#    raise Exception("failed to get checkpoint. going to return ")

isInit = False
tokenizer = None
sess = None
graph = None

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 is_real_example=True):
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

##CLS loss
#def create_model(bert_config, is_training, query_input_ids, query_input_mask, query_segment_ids,
#                 use_one_hot_embeddings):
#    comp_type = tf.float16
#    query_tower_model = modeling.BertModel(
#        config=bert_config,
#        is_training=is_training,
#        input_ids=query_input_ids,
#        input_mask=query_input_mask,
#        token_type_ids=query_segment_ids,
#        use_one_hot_embeddings=use_one_hot_embeddings,
#        comp_type=comp_type,
#        scope="bert")
#
#    query_output_layer = query_tower_model.get_sequence_output()
#    with tf.variable_scope("pos_pooling"):
#        query_first_token_tensor = tf.squeeze(query_output_layer[:, 0:1, :], axis=1)
#
#    return query_first_token_tensor

def create_model(bert_config, is_training, query_input_ids, query_input_mask, query_segment_ids,
                use_one_hot_embeddings):
    comp_type = tf.float32
    query_tower_model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=query_input_ids,
          input_mask=query_input_mask,
          token_type_ids=query_segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings,
          comp_type=comp_type,
          scope="bert")

    query_output_layer = query_tower_model.get_sequence_output()
    with tf.variable_scope("pooling", reuse=tf.AUTO_REUSE):
        query_index_output = tf.layers.dense(
            query_output_layer,
            128,
            kernel_initializer=modeling.create_initializer(0.02))
        query_idx_dropout_ratio = 0.0
        query_output_layer = modeling.dropout(query_index_output, query_idx_dropout_ratio)
        query_output_layer = modeling.layer_norm(query_output_layer)
        hidden_size = query_output_layer.shape[-1]

        hidden_weights = tf.get_variable(
            "bert_weights", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        hidden_bias = tf.get_variable(
            "bert_bias", [1], initializer=tf.zeros_initializer())

        #query_output_layer = query_output_layer[:, 1:-1, :]
        query_output_layer = query_output_layer[:, 1:, :]
        print(tf.expand_dims(hidden_weights, 1))
        query_seq_layer = tf.squeeze(tf.nn.bias_add(tf.matmul(query_output_layer, tf.expand_dims(hidden_weights, 1), transpose_b=True), hidden_bias), axis=2)
        query_attention_score = tf.nn.softmax(query_seq_layer, axis=-1)
        query_first_token_tensor = tf.reduce_sum(tf.expand_dims(query_attention_score, -1) * query_output_layer, axis=1)

    return query_first_token_tensor

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(query_text, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens_a = tokenizer.tokenize(query_text)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    query_tokens = []
    query_segment_ids = []
    query_tokens.append("[CLS]")
    query_segment_ids.append(0)
    for token in tokens_a:
        query_tokens.append(token)
        query_segment_ids.append(0)
    query_tokens.append("[SEP]")
    query_segment_ids.append(0)
    query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)

    query_input_mask = [1] * len(query_input_ids)

    while len(query_input_ids) < max_seq_length:
        query_input_ids.append(0)
        query_input_mask.append(0)
        query_segment_ids.append(0)

    assert len(query_input_ids) == max_seq_length
    assert len(query_input_mask) == max_seq_length
    assert len(query_segment_ids) == max_seq_length

    feature = InputFeatures(
        query_input_ids=query_input_ids,
        query_input_mask=query_input_mask,
        query_segment_ids=query_segment_ids
        )
    return feature


def query_vector_predict(query_info):
    start_time = time.time()
    def convert(examples):
        features_query_input_ids = []
        features_query_input_mask = []
        features_query_segment_ids = []
        for idx, example in enumerate(examples):
            feature = convert_single_example(example, max_seq_length, tokenizer)
            features_query_input_ids.append(feature.query_input_ids)
            features_query_input_mask.append(feature.query_input_mask)
            features_query_segment_ids.append(feature.query_segment_ids)

        real_batch_size = min(batch_size, len(features_query_input_ids))
        if real_batch_size != batch_size:
            filling_query_input_ids = [0] * max_seq_length
            filling_query_input_mask = [0] * max_seq_length
            filling_query_segment_ids = [0] * max_seq_length
            for i in range(batch_size - real_batch_size):
                features_query_input_ids.append(filling_query_input_ids)
                features_query_input_mask.append(filling_query_input_mask)
                features_query_segment_ids.append(filling_query_segment_ids)
        input_ids = np.reshape(features_query_input_ids,(batch_size, max_seq_length))
        input_mask = np.reshape(features_query_input_mask,(batch_size, max_seq_length))
        segment_ids = np.reshape(features_query_segment_ids,(batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids

    global graph
    with graph.as_default():

        query_input_ids, query_input_mask, query_segment_ids = convert(query_info)

        feed_dict = {input_ids: query_input_ids,
                    input_mask: query_input_mask,
                    segment_ids: query_segment_ids}

        pred_query_vec = sess.run([pred_vec], feed_dict)

        time_cost = (time.time() - start_time) * 1000.0
        tf.logging.info("predict time: %f ms per batch" % time_cost)
        #print("predict time: %f ms per batch" % time_cost)
        #print("pred_query_vec length %s" % len(pred_query_vec))
        #print(pred_query_vec)
        #print("pred_query_vec length %s" % len(tf.squeeze(pred_query_vec)))
        return tf.squeeze(np.array(pred_query_vec)).eval(session=sess)
        #return pred_query_vec

def init_embedding_table():
    with tf.variable_scope("embedding_tables", reuse=tf.AUTO_REUSE):
        query_type_embedding_table = tf.get_variable(
            name='query_type',
            shape=[et_query_type_class_num, et_hidden_size],
            initializer=create_initializer(initializer_range=0.02)) #query

        geohash_embedding_table = tf.get_variable(
            name='geohash',
            shape=[et_geohash_class_num, et_hidden_size],
            initializer=create_initializer(initializer_range=0.02)) #poi/query

        week_bkt_embedding_table = tf.get_variable(
            name='week_bucket',
            shape=[et_week_bucket_class_num, et_hidden_size],
            initializer=create_initializer(initializer_range=0.02)) #query

        day_bkt_embedding_table = tf.get_variable(
            name='day_bucket',
            shape=[et_day_bucket_class_num, et_hidden_size],
            initializer=create_initializer(initializer_range=0.02)) #query

    return query_type_embedding_table, geohash_embedding_table, week_bkt_embedding_table, day_bkt_embedding_table

def get_export_model(bert_config_file, use_one_hot_embeddings=False, export_path=model_ckpt):
    builder = tf.saved_model.builder.SavedModelBuilder(model_export_dir)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    model_file = export_path

    with tf.Graph().as_default(), tf.Session(config=tf_config) as tf_sess:
        input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')
        #new
        query_type = tf.placeholder(tf.int32, [None, 1], name='query_type')
        query_geohash_id = tf.placeholder(tf.int32, [None, 1], name='query_geohash_id')
        query_week = tf.placeholder(tf.int32, [None, 1], name='query_week')
        query_day = tf.placeholder(tf.int32, [None, 1], name='query_day')

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        bert_embedding_vec = create_model(bert_config, False, input_ids, input_mask, segment_ids, use_one_hot_embeddings)
        query_pooling_token_tensor_text = tf.expand_dims(bert_embedding_vec, axis=1)

        query_type_embedding_table, geohash_embedding_table, week_bkt_embedding_table, day_bkt_embedding_table = init_embedding_table()
        query_type_tensor = tf.gather(query_type_embedding_table, query_type)
        query_geohash_tensor = tf.gather(geohash_embedding_table, query_geohash_id)
        query_week_bkt_tensor = tf.gather(week_bkt_embedding_table, query_week)
        query_day_bkt_tensor = tf.gather(day_bkt_embedding_table, query_day)
        predict_vec = tf.cast( tf.reduce_mean(tf.concat( \
                [query_pooling_token_tensor_text, query_type_tensor,  query_week_bkt_tensor, query_day_bkt_tensor, query_geohash_tensor], \
                axis=1), axis=1), tf.float64)

        saver = tf.train.Saver()
        print("last_chk:", tf.train.latest_checkpoint(model_file))
        saver.restore(tf_sess, tf.train.latest_checkpoint(model_file))

        predict_vector_export = tf.saved_model.utils.build_tensor_info(predict_vec)
        input_ids_export = tf.saved_model.utils.build_tensor_info(input_ids)
        input_mask_export = tf.saved_model.utils.build_tensor_info(input_mask)
        segment_ids_export = tf.saved_model.utils.build_tensor_info(segment_ids)
        query_type_export = tf.saved_model.utils.build_tensor_info(query_type)
        query_geohash_id_export = tf.saved_model.utils.build_tensor_info(query_geohash_id)
        query_week_export = tf.saved_model.utils.build_tensor_info(query_week)
        query_day_export = tf.saved_model.utils.build_tensor_info(query_day)

        model_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={"input_ids": input_ids_export,
                        "input_mask": input_mask_export,
                        "segment_ids": segment_ids_export,
                        "query_type": query_type_export,
                        "query_geohash_id": query_geohash_id_export,
                        "query_week": query_week_export,
                        "query_day": query_day_export
                        },
                outputs={"predict_vector": predict_vector_export},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )
        builder.add_meta_graph_and_variables(
            tf_sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=tf.group(tf.tables_initializer(), name="legacy_init_op")
        )
        builder.save()


def ExportTest():
    print("========================test using exported serving model=========================")
    tf.reset_default_graph()
    sess = tf.Session()
    export_path = model_export_dir

    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_ids_key = 'input_ids'
    input_mask_key = 'input_mask'
    segment_ids_key = 'segment_ids'
    query_type_key = 'query_type'
    query_geohash_id_key = 'query_geohash_id'
    query_week_key = 'query_week'
    query_day_key = 'query_day'
    predict_vector_key = 'predict_vector'
    print("begin load")
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], export_path)
    print("end load")
    # 从meta_graph_def中取出SignatureDef对象
    signature = meta_graph_def.signature_def

    input_ids_tensor_name = signature[signature_key].inputs[input_ids_key].name
    input_mask_tensor_name = signature[signature_key].inputs[input_mask_key].name
    segment_ids_tensor_name = signature[signature_key].inputs[segment_ids_key].name
    query_type_tensor_name = signature[signature_key].inputs[query_type_key].name
    geohash_id_tensor_name = signature[signature_key].inputs[query_geohash_id_key].name
    query_week_tensor_name = signature[signature_key].inputs[query_week_key].name
    query_day_tensor_name = signature[signature_key].inputs[query_day_key].name
    predict_vector_tensor_name = signature[signature_key].outputs[predict_vector_key].name

    # 获取tensor 并inference
    input_ids = sess.graph.get_tensor_by_name(input_ids_tensor_name)
    input_mask = sess.graph.get_tensor_by_name(input_mask_tensor_name)
    segment_ids = sess.graph.get_tensor_by_name(segment_ids_tensor_name)
    query_type  = sess.graph.get_tensor_by_name(query_type_tensor_name)
    query_geohash_id = sess.graph.get_tensor_by_name(geohash_id_tensor_name)
    query_week = sess.graph.get_tensor_by_name(query_week_tensor_name)
    query_day = sess.graph.get_tensor_by_name(query_day_tensor_name)
    predict_vector = sess.graph.get_tensor_by_name(predict_vector_tensor_name)

    # 测试样例：文本需要经过python函数处理后才能输入bert模型
    #tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    '''输入数据：query, poi_name, poi_address'''
    query_list = ['av euclides figueiredo 965 lama']
    query_type_list = [[1]]
    geohash_list = [[156645]]
    query_week_list = [[4]]
    query_day_list = [[6]]
    FEATURES = []
    for i in range(len(query_list)):
        query = query_list[i]
        query = tokenization.convert_to_unicode(query)
        '''将文本数据转换为feature'''
        feature = convert_single_example(query, max_seq_length, tokenizer)
        FEATURES.append(feature)
    input_1 = []
    input_2 = []
    input_3 = []
    input_4 = []
    input_5 = []
    input_6 = []
    input_7 = []
    for idx, feature in enumerate(FEATURES):
        print(feature.query_input_ids)
        input_1.append(feature.query_input_ids)
        input_2.append(feature.query_input_mask)
        input_3.append(feature.query_segment_ids)
        input_4.append(query_type_list[idx])
        input_5.append(geohash_list[idx])
        input_6.append(query_week_list[idx])
        input_7.append(query_day_list[idx])

    res = sess.run(predict_vector, feed_dict={input_ids: input_1,
                                  input_mask: input_2,
                                  segment_ids: input_3,
                                    query_type: input_4, query_geohash_id: input_5, query_week: input_6, query_day: input_7})

    print("测试样例向量")
    for i in range(len(query_list)):
        query = query_list[i]
        vector = str(res[i].tolist())
        print('\t'.join([query, vector]))


if __name__ == "__main__":
    if not isInit:
        isInit = True
        #tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        '''
        graph = tf.Graph()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        sess=tf.Session(config=gpu_config, graph=graph)
        with graph.as_default():
            print("going to restore checkpoint")
            input_ids = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_ids")
            input_mask = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_mask")
            segment_ids = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="segment_ids")

            bert_config = modeling.BertConfig.from_json_file(bert_config_file)
            pred_vec = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(output_dir))
        '''
    print(vocab_file)
    print(bert_config_file)
    print(model_ckpt)
    print(model_export_dir)
    print(et_geohash_class_num)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    get_export_model(bert_config_file, False, model_ckpt)
    #ExportTest()

