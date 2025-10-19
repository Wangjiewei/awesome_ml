source ./config.sh

set -x

# change_permission_to_poi_data
change_permission_to_map_search

# 生产tfrecord样本所需包
wget -P ../utils https://repo1.maven.org/maven2/org/tensorflow/spark-tensorflow-connector_2.11/1.11.0/spark-tensorflow-connector_2.11-1.11.0.jar

function dict_data_generation(){
    start_yyyymmdd=$1
    end_yyyymmdd=$2
    output_partition=$3
    output_format=$4
    #词表hdfs路径
    dict_file=ES_${start_yyyymmdd}_${end_yyyymmdd}.${output_format}.dict.all
    hdfs_output_path=${hdfs_output_root_path}/ES_${start_yyyymmdd}_${end_yyyymmdd}_dict/
    dict_hdfs_root_path=hdfs://DClusterUS1${hdfs_output_root_path}/
    spark-submit\
        --queue ${yarn_queue} \
        --num-executors 100 \
        --executor-memory 4G \
        --executor-core 2 \
        --conf spark.dynamicAllocation.enabled=true \
        --conf spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursion=true \
        --archives hdfs://DClusterUS1/user/prod_poi_data/poi_data/poi_production_i18n/lib/conda.zip#mypython \
        --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./mypython/bin/python \
        --conf spark.default.parallelism=1000 \
        --files ${dict_hdfs_root_path},../dictionary/norm_char.txt,../dictionary/num_dic.txt,../dictionary/pt_num.txt,../dictionary/stop_words.txt \
        --py-files ../src/common/*,../src/operators/*,../src/config.py \
        ../src/feature_dict_generate.py \
        ${start_yymmdd} \
        ${end_yymmdd} \
        ${country_code_str} \
        ${dump_feature_log_root_path} \
        ${hdfs_output_path} \
        ${output_partition} 

    if [ $? -ne 0 ];then
        echo "dict processing failed"
        exit 1
    fi
    # 本地文件路径
    tmp_file=${local_root_path}/ES-tmp.local.dat
    # 拉本地
    hadoop fs -getmerge ${hdfs_output_path}* ${tmp_file}
    # 去重 & 编码
    python ../src/common/0_convert_dictionary_for_deepfm.py "deepfm" ${tmp_file} ${dict_file}
    # 上传hdfs
    hadoop fs -put ${dict_file} ${hdfs_output_root_path}
    # 清理现场
    rm ${tmp_file}
}

output_format="deepfm"
dict_data_generation ${dict_start_yyyymmdd} ${dict_end_yyyymmdd} ${dict_output_partition} ${output_format} 