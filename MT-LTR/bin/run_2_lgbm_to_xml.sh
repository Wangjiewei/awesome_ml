source ./config.sh

${CONDA_PY_BIN} ../src/lgbm_to_xml.py ${lgbm_output_model} 0
${CONDA_PY_BIN} ../src/lgbm_to_xml.py ${lgbm_output_model} 1