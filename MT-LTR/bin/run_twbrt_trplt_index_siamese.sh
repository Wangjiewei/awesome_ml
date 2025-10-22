#!/bin/bash

work_path=`pwd`
pushd ${work_path}


echo 'current working directory:'${work_path}

BERT_BASE_DIR=${work_path}/../pretrain_lm/electra-base-portuguese-uncased-brwac
DATA_DIR=${work_path}/../samples
OUTPUT_DIR=${work_path}/../models/siamese_mlp_number_local
#OUTPUT_HDFS=hdfs://DClusterUS1/user/prod_poi_data/poi_data/poi_production_i18n/wjw/t6_3_train_tfrecord
#INIT_CKPT=$BERT_BASE_DIR/model.ckpt
#INIT_CKPT=$OUTPUT_DIR/model.ckpt-161053

#/nfs/project/wangjiewei/py37tf15/bin/python run_twbrt_trplt_index_siamese.py \
#/nfs/project/wangjiewei/py37tf15/bin/python run_twbrt_trplt_index_siamese_mlp1.py \
/nfs/volume-100001-10/wangjiewei/py37tf15/bin/python run_twbrt_trplt_index_siamese_20221011.py \
  --do_train=True \
  --do_eval=False \
  --do_predict=False \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --do_lower_case=True \
  --vocab_file=$BERT_BASE_DIR/vocab2.txt \    #开源的西语bert词表
  --bert_config_file=$BERT_BASE_DIR/config.json2 \ #开源的西语bert config
  --init_checkpoint=$INIT_CKPT \
  --max_seq_length=32 \
  --train_batch_size=400 \
  --eval_batch_size=400 \
  --predict_batch_size=400 \
  --learning_rate=2e-5 \
  --num_train_epochs=2 \
  --save_checkpoints_steps=1000 \
  --keep_checkpoint_max=3 \
  --log_name=twbrt_logs.txt \
  --predict_result_name=twbrt_results.tsv \
  --predict_vector_name=twbrt_vectors_poi.tsv

# --init_checkpoint=$INIT_CKPT \
popd