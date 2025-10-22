# coding=utf-8
"""bert finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import collections
import time
import modeling
from inputs import *
import random
import multi_optimization
import tokenization
import tensorflow as tf
import logging
import json
import csv


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = InteractiveSession(config=config)

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

def create_token_pooling_embedding(bert_config, is_training, bert_input_ids, bert_input_mask, bert_segment_ids, use_one_hot_embeddings, comp_type):
    bert_tower_model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=bert_input_ids,
        input_mask=bert_input_mask,
        token_type_ids=bert_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        comp_type=comp_type,
        scope="bert")

    bert_output_layer = bert_tower_model.get_sequence_output()

    
    bert_embedding = bert_tower_model.get_embedding_output()

    with tf.variable_scope("pooling", reuse=tf.AUTO_REUSE):
        bert_index_output = tf.layers.dense(
            bert_output_layer,
            FLAGS.index_size,
            kernel_initializer=modeling.create_initializer(0.02))
        bert_idx_dropout_ratio = 0.0
        if is_training:
            bert_idx_dropout_ratio = 0.2
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
        bert_attention_score = tf.nn.softmax(bert_seq_layer, axis=-1)
        bert_pooling_token_tensor = tf.reduce_sum(tf.expand_dims(bert_attention_score, -1) * bert_output_layer, axis=1)
        

    return bert_pooling_token_tensor

def query_token_pooling_embedding(bert_config, is_training, bert_input_ids, bert_input_mask, bert_segment_ids, use_one_hot_embeddings, comp_type, bert_tower_model):

    # the same embedding_table as poi tower
    embedding_table = bert_tower_model.get_embedding_table()    
   
    with tf.variable_scope("query_pooling", reuse=tf.AUTO_REUSE):
        #embedding_table = tf.get_variable(
        #    name="query_embedding_table",
        #    shape=[bert_config.vocab_size, bert_config.hidden_size],
        #    initializer=tf.truncated_normal_initializer(stddev=0.02))
        # mask is ok
        query_embeddings = tf.nn.embedding_lookup(embedding_table, bert_input_ids)

        bert_index_output1 = tf.layers.dense(
            query_embeddings,
            bert_config.hidden_size,
            kernel_initializer=modeling.create_initializer(0.02))

        bert_index_output2 = tf.layers.dense(
            bert_index_output1,
            bert_config.hidden_size,
            kernel_initializer=modeling.create_initializer(0.02)) 

        bert_index_output = tf.layers.dense(
            bert_index_output2,
            FLAGS.index_size,
            kernel_initializer=modeling.create_initializer(0.02))


        hidden_size = bert_index_output.shape[-1]
        bert_weights = tf.get_variable(
            "query_weights", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        bert_bias = tf.get_variable(
            "query_bias", [1], initializer=tf.zeros_initializer())

        bert_output_layer = bert_index_output[:, 1:, :]
        bert_seq_layer = tf.squeeze(tf.nn.bias_add(tf.matmul(bert_output_layer, bert_weights, transpose_b=True), \
                                                  bert_bias), axis=2)
        bert_attention_score = tf.nn.softmax(bert_seq_layer, axis=-1)
        bert_pooling_token_tensor = tf.reduce_sum(tf.expand_dims(bert_attention_score, -1) * bert_output_layer, axis=1)

    return bert_pooling_token_tensor

def create_model(bert_config, is_training, query_input_ids, query_input_mask, query_segment_ids,
                 pos_input_ids, pos_input_mask, pos_segment_ids, neg_input_ids, neg_input_mask,
                 neg_segment_ids, use_one_hot_embeddings, mode):
    comp_type = tf.float16 if FLAGS.use_fp16 else tf.float32
    
    pos_pooling_token_tensor = create_token_pooling_embedding(bert_config, is_training, pos_input_ids, pos_input_mask, pos_segment_ids, use_one_hot_embeddings, comp_type)

    query_pooling_token_tensor = create_token_pooling_embedding(bert_config, is_training, query_input_ids, query_input_mask, query_segment_ids, use_one_hot_embeddings, comp_type)

    if mode == tf.estimator.ModeKeys.TRAIN \
      or mode == tf.estimator.ModeKeys.EVAL\
      or mode == tf.estimator.ModeKeys.PREDICT:
        neg_pooling_token_tensor = create_token_pooling_embedding(bert_config, is_training, neg_input_ids, neg_input_mask, neg_segment_ids, use_one_hot_embeddings, comp_type)
        with tf.variable_scope("crossing"):
            cosin_similarity_qp = tf.keras.losses.cosine_similarity(query_pooling_token_tensor, \
                                                                    pos_pooling_token_tensor, axis=1)
            cosin_similarity_qp = tf.reshape(cosin_similarity_qp, [-1, 1])
            cosin_similarity_qn = tf.keras.losses.cosine_similarity(query_pooling_token_tensor, \
                                                                    neg_pooling_token_tensor, axis=1)
            cosin_similarity_qn = tf.reshape(cosin_similarity_qn, [-1, 1])

        with tf.variable_scope("loss"):
            constant_one = tf.constant(1.0, dtype=tf.float32)
            distance_qp = constant_one - cosin_similarity_qp
            distance_qn = constant_one - cosin_similarity_qn
            margin = tf.constant(0.2, dtype=tf.float32)
            per_example_loss = tf.maximum(distance_qp - distance_qn + margin, 0)
            loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, cosin_similarity_qp, cosin_similarity_qn, distance_qp, distance_qn, query_pooling_token_tensor,pos_pooling_token_tensor,neg_pooling_token_tensor
    else:
        with tf.variable_scope("crossing"):
            cosin_similarity_qp = tf.keras.losses.cosine_similarity(query_pooling_token_tensor, \
                                                                    pos_pooling_token_tensor, axis=1)
            cosin_similarity_qp = tf.reshape(cosin_similarity_qp, [-1, 1])

        with tf.variable_scope("loss"):
            constant_one = tf.constant(1.0, dtype=tf.float32)
            distance_qp = constant_one - cosin_similarity_qp
        return None, None, cosin_similarity_qp, cosin_similarity_qp, distance_qp, distance_qp, query_pooling_token_tensor,pos_pooling_token_tensor, None

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):

    def model_fn(features, mode):
        query_input_ids = features["query_input_ids"]
        query_input_mask = features["query_input_mask"]
        query_segment_ids = features["query_segment_ids"]
        pos_input_ids = features["pos_input_ids"]
        pos_input_mask = features["pos_input_mask"]
        pos_segment_ids = features["pos_segment_ids"]
        neg_input_ids = features["neg_input_ids"]
        neg_input_mask = features["neg_input_mask"]
        neg_segment_ids = features["neg_segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, cosin_similarity_qp, cosin_similarity_qn, distance_qp, distance_qn, query_pooling_token_tensor,pos_pooling_token_tensor,neg_pooling_token_tensor) = \
          create_model(bert_config, is_training, query_input_ids, query_input_mask, query_segment_ids, \
                pos_input_ids, pos_input_mask, pos_segment_ids, neg_input_ids, neg_input_mask, \
                neg_segment_ids, use_one_hot_embeddings, mode)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = multi_optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss):
                loss = tf.metrics.mean(values=per_example_loss)
                return {
                    "eval_loss": loss
                }
            eval_metric_ops = (metric_fn, [per_example_loss])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metric_ops[0](*eval_metric_ops[1]))
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                          "cosin_similarity_qp": cosin_similarity_qp,
                          "cosin_similarity_qn": cosin_similarity_qn,
                          "distance_qp": distance_qp,
                          "distance_qn": distance_qn,
                          "query_pooling_token_tensor": query_pooling_token_tensor,
                            "pos_pooling_token_tensor": pos_pooling_token_tensor,
                            "neg_pooling_token_tensor": neg_pooling_token_tensor,})
        return output_spec
    return model_fn

def serving_input_fn():
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {"regression": RegressionProcessor}
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict :
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    # multi_gpu
    #devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5'] 
    #devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
    #strategy = tf.distribute.MirroredStrategy(devices=devices, cross_device_ops=tf.distribute.ReductionToOneDevice())
    #strategy = tf.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        train_distribute=None,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max)

    tf.logging.info("***** get_train_examples *****")
    #train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    train_file = None
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.data_dir, FLAGS.train_tf_record_file)
        if not os.path.exists(train_file):
            train_examples_num = processor.gen_tfrecord_file(FLAGS.data_dir, FLAGS.max_seq_length, tokenizer, train_file, "train")
            #file_based_convert_examples_to_features(train_examples, FLAGS.max_seq_length, tokenizer, train_file)
        else:
            train_examples_num = -1
            for count,line in enumerate(open(os.path.join(FLAGS.data_dir, "train.tsv"),'r')):
                pass
            train_examples_num = count + 1
        #train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int( train_examples_num / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # use GPU estimator instead of TPUEstimator to show training process and loss log
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=None
    )

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            batch_size=FLAGS.train_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        TIME = time.time()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        # hooks = [early_stopping_hook])
        TIME = (time.time() - TIME) * 1000.0 / num_train_steps
        tf.logging.info("train time: %f ms per batch" % TIME)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        eval_file = os.path.join(FLAGS.data_dir, FLAGS.dev_tf_record_file)
        if not os.path.exists(eval_file):
            file_based_convert_examples_to_features(eval_examples, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("eval examples size = %d, Batch size = %d", num_actual_eval_examples, \
                                  FLAGS.eval_batch_size)
        eval_steps = None
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            batch_size=FLAGS.eval_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        stime = time.time()
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        csmtime = (time.time() - stime) * 1000.0 / (num_actual_eval_examples // int(FLAGS.eval_batch_size))
        tf.logging.info("valid time: %f ms per batch" % csmtime)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        predict_file = os.path.join(FLAGS.data_dir, FLAGS.test_tf_record_file)
        if not os.path.exists(predict_file):
            file_based_convert_examples_to_features(predict_examples, FLAGS.max_seq_length, \
                                                    tokenizer, predict_file)
        tf.logging.info("Batch size = %d", FLAGS.predict_batch_size)
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            batch_size=FLAGS.predict_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        stime = time.time()
        #result = estimator.predict(input_fn=predict_input_fn)
        #output_predict_file = os.path.join(FLAGS.output_dir, FLAGS.predict_result_name)
        #with tf.io.gfile.GFile(output_predict_file, "w") as pred_writer:
        #    num_written_lines = 0
        #    tf.logging.info("***** predict results *****")
        #    for (i, (example, prediction)) in enumerate(zip(predict_examples, result)):
        #        cosine_qp = prediction["cosin_similarity_qp"]
        #        cosine_qn = prediction["cosin_similarity_qn"]
        #        if i % 500000 == 0:
        #            tf.logging.info("i-th : %s \t prediction : %s" % (i, prediction))
        #            tf.logging.info("predict_process:%d / %d" % (i, num_actual_predict_examples))
        #        information = [example.text_a, example.text_b, example.text_c, str(cosine_qp[0]), str(cosine_qn[0])]
        #        output_line = '\t'.join(information) + '\n'
        #        pred_writer.write(output_line)
        #        num_written_lines += 1
        #    cstime = (time.time() - stime) * 1000.0 / ((num_actual_predict_examples + FLAGS.predict_batch_size - 1) // int(FLAGS.predict_batch_size))
        #    tf.logging.info("predict time: %f ms per batch" % cstime)
        #    tf.logging.info("num_written_lines:%d", num_written_lines)
        #    tf.logging.info("num_actual_predict_examples: %d", num_actual_predict_examples)

        result = estimator.predict(input_fn=predict_input_fn)
        #for i in predict_input_fn:
        #    print(i)
        #estimator._export_to_tpu = False
        #estimator.export_savedmodel(os.path.join(FLAGS.output_dir, "1"), serving_input_fn)
        output_vector_file = os.path.join(FLAGS.output_dir, FLAGS.predict_vector_name)
        tf.logging.info("output_vector_file: %s " % output_vector_file)
        with tf.io.gfile.GFile(output_vector_file, "w") as pvec_writer:
            num_written_lines = 0
            query_set = set()
            doc_set = set()
            tf.logging.info("***** predict vectors *****")
            for (i, (example, prediction)) in enumerate(zip(predict_examples, result)):
                query_pooling_tensor = prediction["query_pooling_token_tensor"]
                pos_pooling_tensor = prediction["pos_pooling_token_tensor"]
                neg_pooling_tensor = prediction["neg_pooling_token_tensor"]
                #information = [example.text_a, example.text_b, example.text_c, str(cosine_qp[0]), str(cosine_qn[0])]
                if i % 500000 == 0:
                    tf.logging.info("predict_process:%d / %d" % (i, num_actual_predict_examples))
                
                if(example.text_a not in query_set):
                    #query_set.add(example.text_a)
                    vecinfo = ['query',example.text_a, example.poiid_b, ",".join(str(e) for e in query_pooling_tensor), example.cityid_b]
                    output_line = '\t'.join(vecinfo) + '\n'
                    pvec_writer.write(output_line)
                if(example.text_b not in doc_set):
                    #doc_set.add(example.text_b)
                    vecinfo = ['doc',example.text_b, example.poiid_b, ",".join(str(e) for e in pos_pooling_tensor), example.cityid_b]
                    output_line = '\t'.join(vecinfo) + '\n'
                    pvec_writer.write(output_line)
                if(example.text_c not in doc_set):
                    #doc_set.add(example.text_c)
                    vecinfo = ['doc',example.text_c, example.poiid_c, ",".join(str(e) for e in neg_pooling_tensor), example.cityid_c]
                    output_line = '\t'.join(vecinfo) + '\n'
                    pvec_writer.write(output_line)

                num_written_lines += 1
            tf.logging.info("num_written_lines:%d", num_written_lines)  
                

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")

    # log file settings
    tf.io.gfile.makedirs(FLAGS.output_dir)
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(FLAGS.output_dir, FLAGS.log_name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    tf.app.run()
