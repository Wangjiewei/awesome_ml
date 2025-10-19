source ./config.sh

echo  "######## lgbm train begin ########"
echo ${train_config_file}
echo ${train_local_file}
echo ${valid_local_file}
echo ${lgbm_output_model}
${LGBM_BIN} \
    config=${train_config_file} \
    train_data=${train_local_file} \
    valid_data=${valid_local_file} \
    output_model=${lgbm_output_model}
echo "########## lgbm train end ##########"

echo "######### lgbm predict begin #########"
echo 




${LGBM_BIN} \
    config=${valid_config_file} \
    input_model=${lgbm_output_model} \
    data=${valid_local_file} \
    output_result=${predict_result}
echo "########## lgbm predict end ##########"

# 计算top1/3/5点击率
# paste 

OIFS=$IFS
IFS=','
for country in ${country_code_str}
do 
    echo ${country}
    c_valid_local_file=${local_root_path}/${country}-valid_${valid_start_yymmdd}-${valid_end_yymmdd}.l2format
    c_predict_result=../model/${country}.lgbm.predict.res
    c_evaluate_local_file=${c_valid_local_file}.evaluate
    ${LGBM_BIN} \
        config=${valid_config_file} \
        input_model=${lgbm_output_model} \
        data=${c_valid_local_file} \
        output_result=${c_predict_result}

    paste ${c_evaluate_local_file} ${c_predict_result} | cut -f 1,3,4 | python ../src/lgbm_predict_valid.py
done