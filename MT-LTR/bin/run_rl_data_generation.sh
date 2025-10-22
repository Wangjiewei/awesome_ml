source ./config.sh

# set -x

#change_permission_to_poi_data
change_permission_to_map_search

# 生产tfrecord样本所需包
#wget -P ../utils https://repo1.maven.org/maven2/org/tensorflow/spark-tensorflow-connector_2.11/1.11.0/spark-tensorflow-connector_2.11-1.11.0.jar

# 生产tfrecord样本所需包
#wget -P ../utils https://repo1.maven.org/maven2/org/tensorflow/spark-tensorflow-connector_2.11/1.11.0/spark-tensorflow-connector_2.11-1.11.0.jar

function deepfm_data_generation(){

    start_yyyymmdd=$1
    end_yyyymmdd=$2
    output_partitions=$3
    file_type=$4
    output_format=$5
    vocab_file=$6
    poiidx_hdfs_path=$7
    dict_file="test"
    #poiidx_hdfs_path=hdfs://DClusterUS1//user/prod_poi_data/poi_data/wugaoyin/ltr_exp/poi_idxs
    #poiidx_hdfs_path=hdfs://DClusterUS1//user/prod_poi_data/poi_data/wugaoyin/ltr_exp/br_poi_idx
    #hdfs_input_path=${hdfs_output_root_path}/train_MX.tsv
    #hdfs_input_path=/user/prod_poi_data/poi_data/wugaoyin/ltr_exp/test_query.tsv
    #hdfs_input_path=/user/prod_poi_data/poi_data/wugaoyin/ltr_exp/new_poi.dat
    #hdfs_input_path=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/multifeature_BR202308_withAC_order
    #hdfs_input_path_2=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/multifeature_BR202309_withAC_order
    hdfs_input_path=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/multifeature_MX202308_withAC_order_withoutAC
    hdfs_input_path_2=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/multifeature_MX202309_withAC_order_withouAC
    #hdfs_input_path=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/new_no_hardneg_BR202305_withAC_order
    #hdfs_input_path_2=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/new_no_hardneg_BR202306_withAC_order
    #hdfs_input_path=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/multifeature_MX202308_withAC_poi
    #hdfs_input_path=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/new_no_hardneg_BR202305_withAC_poi
    #hdfs_input_path=/user/prod_poi_data/poi_data/poi_production_i18n/wjw/multifeature_BR202308_withAC_poi
    #hdfs_input_path_2=""
    hdfs_output_path=${hdfs_output_root_path}/${country_code}_${start_yyyymmdd}_${end_yyyymmdd}_${file_type}
    spark-submit\
        --queue ${yarn_queue} \
        --num-executors 100 \
        --executor-memory 10G \
        --executor-cores 2 \
        --conf spark.dynamicAllocation.enabled=true\
        --conf spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive=true \
        --conf spark.default.parallelism=1000\
        --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./mypython/zhiming_conda/bin/python\
        --archives hdfs://DClusterUS1/user/prod_poi_data/poi_data/wugaoyin/ltr_exp/zhiming_conda.zip#mypython\
        --files ../dictionary/norm_char.txt,../dictionary/num_dict.txt,../dictionary/pt_num.txt,../dictionary/stop_words.txt,../dictionary/bert_token_vocab_br,../dictionary/bert_token_vocab_mx \
        --py-files ../src/common/*,../src/operators/*,../src/config.py \
        --jars ../utils/spark-tensorflow-connector_2.11-1.11.0.jar \
        ../src/data_processing_ebr.py \
        ${start_yyyymmdd} \
        ${end_yyyymmdd} \
        ${country_code} \
        ${hdfs_input_path} \
        ${hdfs_output_path} \
        10 \
        ${output_format} \
        "${poiidx_hdfs_path}" \
        ${vocab_file} \
        "${dict_file}" \
        "${hdfs_input_path_2}"
        
        #../src/data_processing.py \

    if [ $? -ne 0 ]; then
        echo "deepfm data processing fail"
        exit 1
    fi
}


#vocab_file="vocab.txt"
vocab_file="bert_token_vocab_mx"
## 生成poi 对应的token ids
outformat="poi"
#file_type=rl_train_poi_tokens_br
file_type=rl_train_poi_tokens_mx_noac
poiid_idx=""

## 生成 训练数据, 需指定 poiid_idx
poiid_idx=hdfs://DClusterUS1//user/prod_poi_data/poi_data/poi_production_i18n/wjw/ltr_exp/mx_poi_idx_202308_safe_noac
#outformat="multy_feature"
outformat="query"
file_type=rl_data_tfrecord_mx_l_safe_noac

deepfm_data_generation ${train_start_yyyymmdd} ${train_end_yyyymmdd} ${train_output_partitions} ${file_type} ${outformat} ${vocab_file} ${poiid_idx}
