# curl -X POST -H "Content-type: application/json" -d \
# "{ \
# \"source_data\" : \"/mnt/mainz01eus/projects/KNNMT_CT/data/wv_ct_data/vw_1m/ende/train.en-de.de\", \
# \"target_data\" : \"/mnt/mainz01eus/projects/KNNMT_CT/data/wv_ct_data/vw_1m/ende/train.en-de.en\", \
# \"lang_dir\" : \"deen\", \
# \"domain_id\" : \"vw_1m\" \
# }" "http://127.0.0.1:12346/index"



curl -X POST -H "Content-type: application/json" -d \
"{ \
\"source_data\" : \"/mnt/mainz01eus/projects/KNNMT_CT/data/taus_ecommerce/raw/ecommerce.en-de.train.deu.snt\", \
\"target_data\" : \"/mnt/mainz01eus/projects/KNNMT_CT/data/taus_ecommerce/raw/ecommerce.en-de.train.enu.snt\", \
\"lang_dir\" : \"deen\", \
\"domain_id\" : \"taus_ecommerce\" \
}" "http://127.0.0.1:12346/index"