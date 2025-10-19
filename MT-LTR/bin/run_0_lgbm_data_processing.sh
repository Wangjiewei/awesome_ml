source ./config.sh

set -x

change_permission_to_poi_data
# change_permission_to_map_search

## EBR模型和词表（第一次上线需要）
#hadoop fs -get /user/prod_poi_data/poi_data/ltr_exp/MX_ebr_l2_ranker/bert_token_vocab_2 ./vocab.txt
#hadoop fs -get /user/map_search/sug_global/raw/luban_model/vector_search/2 ./
#zip -r 1.zip 2
#hadoop fs -put 1.zip /user/prod_poi_data/poi_data/ltr_exp/MX_ebr_l2_ranker

function data_processing(){
    start_yymmdd=$1
    end_yymmdd=$2
    output_partition=$3
    local_file=$4
    country_code=$5
    output_format="gbdt"
    # 生产训练数据（MX的EBR特征已经全量了，后面不需要拼）
    if [ "$country_code" = "MX" ];then
        hdfs_output_path=${hdfs_output_root_path}/${country_code}_${start_yymmdd}_${end_yymmdd}/
    else
        hdfs_output_path=${hdfs_output_root_path}/${country_code}_${start_yymmdd}_${end_yymmdd}_raw/
    fi 
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
        --files ../dictionary/norm_char.txt,../dictionary/num_dic.txt,../dictionary/pt_num.txt,../dictionary/stop_words.txt \
        --py-files ../src/common/*,../src/operators/*,../src/config.py \
        ../src/data_processing.py \
        ${start_yymmdd} \
        ${end_yymmdd} \
        ${country_code} \
        ${dump_feature_log_root_path} \
        ${hdfs_output_path} \
        ${output_partition} \
        ${output_format}    \

        if [ $? -ne 0 ];then
            echo "data_processing failed for ${country_code} from ${start_yymmdd} to ${end_yymmdd}"
            exit 1
        fi

}

function ebr_feat_processing(){













































}


## 1.清空文件
train_tmp_file=${local_root_path}/ES-${train_start_yymmdd}-tmp.local.dat
cat /dev/null > $train_tmp_file

OIFS=$IFS
IFS=','
for country in ${country_code_str}
do 
    # 2.生产
    ## Train数据
    echo 'processing train data'
    data_processing 

    tmp_file=${local_root_path}/${country}_${train_start_yymmdd}-tmp.local.dat
    ## 拉本地
    # 格式: parse_info \t searchid||uid \t group_len \t label \t show_pos \t idx1:feat1 \t idx2:feat2 \t ...
    hdfs_output_path=${hdfs_output_root_path}/${country}_${train_start_yymmdd}_${train_end_yymmdd}
    hadoop fs -getmerge ${hdfs_output_path} ${tmp_file}
    ## 生成三份文件
    # 1） groupId \t gropu_len \t label （用于计算top1/3/5）
    cat ${tmp_file} | cut -f 2-4 >> ${train_local_file}.evaluate
    # 2) group_len (训练)
    cat ${train_local_file}.evaluate | cut -f 1-2 | uniq | cut -f 2 > ${train_local_file}.query
    # 3) label \t feature(训练)
    cat ${tmp_file} | cut -f 4,6- >> ${train_local_file}



done






##### 3.本地处理train data 生成三份文件
##### 1） groupId \t gropu_len \t label （用于计算top1/3/5）
##### cat ${train_tmp_file} | cut -f 2-4 >> ${train_local_file}.evaluate
##### 2) group_len (训练)
##### cat ${train_local_file}.evaluate | cut -f 1-2 | uniq | cut -f 2 > ${train_local_file}.query
##### 3) label \t feature(训练)
##### cat ${train_tmp_file} | cut -f 4,6- >> ${train_local_file}
##### # 清理现场
##### rm ${train_tmp_file}
#####
#####
##### #【注意】需要分国家评估，这里手动改tmp_file的国家前缀， 默认MX
##### 4.本地处理valid data 生成三份文件
##### tmp_file=${local_root_path}/MX-${valid_start_yymmdd}-tmp.local.dat
##### # 分成三份文件
##### 1） groupId \t gropu_len \t label （用于计算top1/3/5）
##### cat ${tmp_file} | cut -f 2-4 >> ${valid_local_file}.evaluate
##### 2) group_len (训练)
##### cat ${valid_local_file}.evaluate | cut -f 1-2 | uniq | cut -f 2 > ${valid_local_file}.query
##### 3) label \t feature(训练)
##### cat ${tmp_file} | cut -f 4,6- > ${valid_local_file}
##### # 清理现场
##### rm ${tmp_file}
#####
#####

