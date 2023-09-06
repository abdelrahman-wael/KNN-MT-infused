K=1
BEAM=1
OUTPUT_PATH=/home/aiscuser/CT/sequential_knnmt/logs/bsz_1_sentence/flat_l2_without_logging
MODEL_PATH=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/training_dir/checkpoints_deen_pretrained/checkpoint_best.pt
LOG_PATH=${OUTPUT_PATH}/logs
DATA_STORES=/mnt/mainz01eus/v-enarouz/data/ELASTICSEARCH/de_en/data_store_de_en_2500_test_elastic_search_and_edit_distance_top_128_32
TEST_TGT_PATH=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/data/CT_vw/wv_ct_data/vw/ende/test/test.en-de.en
dstore_training_data_folder=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/chuncked_data/ds
test_data_folder=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/chuncked_data/de_chunks

mkdir -p ${OUTPUT_PATH}

time CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /home/aiscuser/CT/sequential_knnmt/KNN-MT/fairseq_cli/generate.py /home/ababouelenin/dump \
    --adaptive-temperature 100 \
    --test-data-folder $test_data_folder \
    --dstore-training-data-folder $dstore_training_data_folder \
    --max-tokens 4096 \
    --save-knn-dstore \
    --gen-subset train \
    --knn-add-to-idx \
    --path ${MODEL_PATH} \
    --bpe sentencepiece \
    --beam $BEAM \
    --remove-bpe \
    --tokenizer moses \
    --moses-source-lang de \
    --moses-target-lang en \
    --scoring sacrebleu \
    --knnmt \
    --k $K \
    --probe 32 \
    --indexfile $DATA_STORE/index_only.4096.index.vw.0 \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --knn-keytype last_ffn_input \
    --knn-embed-dim 512 \
    --no-load-keys  \
    --knn-temp 10 \
    --knn-sim-func do_not_recomp_l2 \
    --lmbda 0.8 \
    --use-faiss-only > ${LOG_PATH}

# grep D-0 /home/ababouelenin/dump/log | awk '{$1=$2=""; print $0}' | sacrebleu $TEST_TGT_PATH > $OUTPUT_PATH/scores.txt

# cat sacrebleu $TEST_TGT_PATH > $OUTPUT_PATH/scores.txt
# cat /home/aiscuser/output/1m_knn_output_bsz_20.txt | sacrebleu /mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/data/CT_vw/wv_ct_data/vw/ende/test/test.en-de.en