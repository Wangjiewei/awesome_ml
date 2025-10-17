source ./config.sh

echo  "lgbm train"
echo
echo
echo
echo
${LGBM_BIN} \
    config=${train_config_file} \
    train_data=${train_local_file} \
    valid_data=${valid_local_file} \
    output_model=${lgbm_output_model}

${LGBM_BIN} \
    config=${valid_config_file} \
    input_model=${lgbm_output_model} \
    data=${valid_local_file} \
    output_result=${predict_result}