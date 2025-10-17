source ./config.sh

set -x

change_permission_to_poi_data
# change_permission_to_map_search

## EBR模型和词表（第一次上线需要）
#
#
#
#

function data_processing(){
    start_yymmdd=$1
    end_yymmdd=$2
    output_partition=$3
    local_file=$4
    country_code=$5
    output_format="gbdt"
    # 
    if [ "$country_code" = "MX" ];then
        hdfs_output_path=
    else
        hdfs_output_path=
    fi 
    spark-submit\
        --queue ${yarn_queue} \
        --num-executors 100 \
        --executor-memory 4G \
        --executor-core 2 \
        --conf spark \
        --conf \
        --archives \
        --conf \
        --conf \
        --files \
        --py-files \
        ../src/data_processing.py












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
    echo ''
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

