source ./config.sh

set -x

change_permission_to_poi_data
#change_permission_to_map_search

# 生产tfrecord样本所需包
wget -P ../utils https://repo1.maven.org/maven2/org/tensorflow/spark-tensorflow-connector_2.11/1.11.0/spark-tensorflow-connector_2.11-1.11.0.jar

function get_session_info(){


























}

function click_feature_all(){
    change_permission_to_poi_data
    start_yymmdd=$1
    end_yymmdd=$2
    output_partitions=$3
    click_file_suffix=$4
    session_root_path=$5
    country_code=$6
    #生产训练/验证数据
    hdfs_output_path=${hdfs_output_root_path}/${country_code}_${start_yymmdd}_${end_yymmdd}_${click_file_suffix}
    spark-submit\
        --conf spark.executorEnv.PYTHON_EGG_CACHE="/tmp/.python-eggs/" \
        --conf spark.executorEnv.PYTHON_EGG_DIR="/tmp/.python-eggs/" \
        --conf spark.driverEnv.PYTHON_EGG_CACHE="/tmp/.python-eggs/" \
        --conf spark.driverEnv.PYTHON_EGG_DIR="/tmp/.python-eggs/" \
        --queue ${yarn_queue} \
        --deploy-mode cluster \
        --num-executors 200 \
        --executor-memory 12G \
        --executor-core 1 \
        --conf spark.dynamicAllocation.enabled=true \
        --conf spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursion=true \
        --archives hdfs://DClusterUS1/user/prod_poi_data/poi_data/poi_production_i18n/lib/conda.zip#mypython \
        --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./mypython/bin/python \
        --conf spark.default.parallelism=1000 \
        --files ../featureAll/dictionary/norm_char.txt,../featureAll/dictionary/num_dic.txt,../featureAll/dictionary/pt_num.txt,../featureAll/dictionary/stop_words.txt \
        --py-files ../featureAll/QueryClickOrder.py,../featureAll/Requirement.py,../featureAll/utils.py,../src/common/*,../src/operators/*,../src/config.py \
        ../featureAll/genFeature.py \
        ${start_yymmdd} \
        ${end_yymmdd} \
        ${country_code} \
        ${hdfs_output_path} \
        ${output_partition} \
        ${session_root_path}    

        if [ $? -ne 0 ];then
            echo "click_feature_all failed for ${country_code} from ${start_yymmdd} to ${end_yymmdd}"
            exit 1
        fi
}


function mtr_data_generation(){
    change_permission_to_poi_data
    # 入参
    start_yyyymmdd=$1
    end_yyyymmdd=$2
    output_partitions=$3
    file_suffix=$4
    click_file_suffix=$5
    file_type=$6
    country_code=$7
    output_format=$8
    # 词表路径
    dict_file=ES_${dict_start_yyyymmdd}_${dict_end_yyyymmdd}.deepfm.dict.all
    dict_file_path=hdfs://DClusterUS1${hdfs_output_root_path}/${dict_file}
    # click featureAll数据hdfs路径
    click_featureAll_hdfs_path=${hdfs_output_root_path}/${country_code}_${start_yyyymmdd}_${end_yyyymmdd}_${click_file_suffix}
    # 生产训练/验证数据
    hdfs_output_path=${hdfs_output_root_path}/${country_code}_${start_yyyymmdd}_${end_yyyymmdd}_${file_suffix}/
    spark-submit\
        --queue ${yarn_queue} \
        --num-executors 200 \
        --executor-memory 12G \
        --executor-core 1 \
        --conf spark.dynamicAllocation.enabled=true \
        --conf spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursion=true \
        --archives hdfs://DClusterUS1/user/prod_poi_data/poi_data/poi_production_i18n/lib/conda.zip#mypython \
        --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./mypython/bin/python \
        --conf spark.default.parallelism=1000 \
        --files ${dict_hdfs_path},${lgbm_output_model},../dictionary/norm_char.txt,../dictionary/num_dic.txt,../dictionary/pt_num.txt,../dictionary/stop_words.txt \
        --py-files ../src/common/*,../src/operators/*,../src/config.py \
        --jars ../utils/spark-tensorflow-connector_2.11-1.11.0.jar \
        ../src/data_processing.py \
        ${start_yyyymmdd} \
        ${end_yyyymmdd} \
        ${country_code} \
        ${dump_feature_log_root_path} \
        ${hdfs_output_path} \
        ${output_partition} \
        ${output_format} \
        ${dict_file} \
        ${gbdt_model_name} \
        ${click_featureAll_hdfs_path} \
        ${file_type}

    if [ $? -ne 0 ];then
        echo "mtr data processing failed"
        exit 1
    fi
}

# 全量之后不需要手动拼接EBR特征
function ebr_feat_processing(){
    



































}

function multicountry_data_shuffle(){























}


OIFS=$IFS
IFS=','
for country in ${country_code_str}
do
    #step0: 
    #kuang_time_root_path
    kuang_time_root_path=/user/map_search/sug_global/raw/es_mtr_online_learn/es_mtr_exp
    echo 'processing click session train data'
    get_session_info ${train_start_yyyymmdd} ${train_end_yyyymmdd} ${kuang_time_root_path} ${country}

    echo 'processing click session valid data'
    get_session_info ${valid_start_yyyymmdd} ${valid_end_yyyymmdd} ${kuang_time_root_path} ${country}


    #step1: 回溯














    # step2: 生成MTR数据
    output_format="mtr"
    # 训练集
    file_suffix=mtr_tfrecord_all
    file_type=train
    echo 'processing mtr train data'
    mtr_data_generation ${train_start_yyyymmdd} ${train_end_yyyymmdd} ${train_output_partition} ${file_suffix} ${click_file_suffix} ${file_type} ${country} ${output_format}

    # 训练用的验证集
    file_suffix=mtr_tfrecord_valid_all
    file_type=valid
    echo 'processing mtr valid data'
    mtr_data_generation ${valid_start_yyyymmdd} ${valid_end_yyyymmdd} ${valid_output_partition} ${file_suffix} ${click_file_suffix} ${file_type} ${country} ${output_format}

    # 测试集
    file_suffix=mtr_tfrecord_all
    file_type=test
    echo 'processing mtr test data'
    mtr_data_generation ${valid_start_yyyymmdd} ${valid_start_yyyymmdd} ${valid_output_partition} ${file_suffix} ${click_file_suffix} ${file_type} ${country} ${output_format}

done
IFS=$OIFS
unset OIFS


# step3: 多国数据打乱
# 训练集
file_suffix=mtr_tfrecord_all
input_path=${hdfs_output_root_path}/*_${train_start_yyyymmdd}_${train_end_yyyymmdd}_${file_suffix}
output_path=${hdfs_output_root_path}/ES_${train_start_yyyymmdd}_${train_end_yyyymmdd}_${file_suffix}
multicountry_data_shuffle ${input_path} ${output_path}

# 训练用的验证集
file_suffix=mtr_tfrecord_valid_all
input_path=${hdfs_output_root_path}/*_${valid_start_yyyymmdd}_${valid_end_yyyymmdd}_${file_suffix}
output_path=${hdfs_output_root_path}/ES_${valid_start_yyyymmdd}_${valid_end_yyyymmdd}_${file_suffix}
multicountry_data_shuffle ${input_path} ${output_path}
