## 账号和队列 ##
change_permission_to_poi_data(){
    export HADOOP_USER_NAME=prod_poi_data
    export HADOOP_USER_PASSWORD=Poi@data2020
    yarn_queue=root.map_poi_data_prod
}

## ltr特征现场日志集群路径
dump_feature_log_root_path=/user/prod_poi_data/poi_data/poi_production_i18n/genFeature

#### 修改下面的参数适配训练流程
##
country_code_arr=(MX CO CL)
country_code_str=$(IFS=, ; echo "${country_code_arr[*]}")

## train/valid/dict时间范围
train_backtrace_days=30
train_start_yyyymmdd=20240923
train_end_yyyymmdd=20241023
train_output_partition=200

valid_backtrace_days=7
valid_start_yyyymmdd=20241024
valid_end_yyyymmdd=20241030
valid_output_partition=200

dict_backtrace_days=60
dict_start_yyyymmdd=20240830
dict_end_yyyymmdd=20241030
dict_output_partition=20

## hdfs数据产出路径
hdfs_output_root_path=/user/prod_poi_data/poi_data/poi_production_i18n/es_mtr_exp_online


## local数据存储路径
local_root_path=../data
if [ ! -d ${local_root_path} ];then
    mkdir -p ${local_root_path}
fi

## lgbm本地数据路径
train_local_file=${local_root_path}/ES-${train_start_yyyymmdd}-${train_end_yyyymmdd}.l2format
valid_local_file=${local_root_path}/ES-${valid_start_yyyymmdd}-${valid_end_yyyymmdd}.l2format
evaluate_local_file=${valid_local_file}.evaluate

## lgbm环境和conda_py环境
lgbm_bin=/home/odin/map_search/common/LightGBM/lightgbm
conda_py_bin=/home/odin/map_search/common/conda/bin/python
LGBM_BIN=${lgbm_bin}
CONDA_PY_BIN=${conda_py_bin}

## lgbm模型配置
train_config_file=../config/ES.gbdt_train.conf
valid_config_file=../config/ES.gbdt_valid.conf
model_version=v3.0_mtr_es

## lgbm模型文件 & 预测结果
gbdt_model_name=ES.lgbm.${model_version}
lgbm_output_model=../model/${gbdt_model_name}
predict_result=../model/ES.lgbm.predict.res


