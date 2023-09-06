K=1
BEAM=1
# OUTPUT_PATH=/home/ababouelenin/CustomTranslation/sequential_KNNMT/bash_script
MODEL_PATH=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/training_dir/checkpoints_deen_pretrained/checkpoint_best.pt
# MODEL_PATH=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/training_dir/ckpt_deen_taus_ecommerce/checkpoint_best_1.pt
# MODEL_PATH=/mnt/mtcairogroup/Users/mohamedmaher/CTv2/KNN/fairseq_training/training/train_en-cs_lr5-4_WU4000_MT4096/checkpoint65.pt
MODEL_PATH=/mnt/batch/tasks/shared/LS_root/mounts/clusters/masterthesis/code/Users/abdelrahman.abouelenin/checkpoint_best.pt
# LOG_PATH=${OUTPUT_PATH}/logs
# DATA_STORES=/mnt/mainz01eus/v-enarouz/data/ELASTICSEARCH/de_en/data_store_de_en_2500_test_elastic_search_and_edit_distance_top_128_32
# TEST_TGT_PATH=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/data/CT_vw/wv_ct_data/vw/ende/test/test.en-de.en
dstore_training_data_folder=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/chuncked_data/ds
test_data_folder=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/chuncked_data/de_chunks
# LANG_PAIRS="zh-en,en-zh,en-ja,ja-en,ru-en,en-ru,de-en,en-de"
LANG_PAIRS="de-en,en-de,en-fr,fr-en"
LANG_LIST="./lang_list"

CUDA_VISIBLE_DEVICES=0 python /mnt/batch/tasks/shared/LS_root/mounts/clusters/masterthesis/code/Users/abdelrahman.abouelenin/KNN-MT/flask_server.py ./vocab \
    --adaptive-temperature 100 \
    --test-data-folder $test_data_folder \
    --dstore-training-data-folder $dstore_training_data_folder \
    --task translation_multi_simple_epoch \
    --sentencepiece-model ./vocab/spm.model \
    --max-tokens 8126 \
    --save-knn-dstore \
    --lang-dict $LANG_LIST \
    --lang-pairs $LANG_PAIRS \
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
    --quiet \
    --k $K \
    --probe 32 \
    --no-progress-bar \
    --indexfile $DATA_STORE/index_only.4096.index.vw.0 \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --knn-keytype last_ffn_input \
    --no-load-keys  \
    --knn-temp 10 \
    --decoder-langtok \
    --knn-sim-func do_not_recomp_l2 \
    --skip-invalid-size-inputs-valid-test \
    --knn-embed-dim 1536 \
    --lmbda 0.8 \
    --use-faiss-only 

    # --max-sentences 100 \
    # --knn-embed-dim 512 \

# grep D-0 /home/ababouelenin/dump/log | awk '{$1=$2=""; print $0}' | sacrebleu $TEST_TGT_PATH > $OUTPUT_PATH/scores.txt



# "--adaptive-temperature" ,"100",
# "--test-data-folder","/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/chuncked_data/de_chunks",
# "--dstore-training-data-folder","/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/chuncked_data/ds",
# "--task","translation_multi_simple_epoch",
# "--max-tokens","4096",
# "--save-knn-dstore",
# "--lang-dict ,"$LANG_LIST",
# "--lang-pairs ,"$LANG_PAIRS",
# "--gen-subset ,"train",
# "--knn-add-to-idx",
# "--path","",
# "--bpe" ,"sentencepiece",
# "--beam","1",
# "--remove-bpe",
# "--tokenizer" ,"moses",
# "--moses-source-lang","de",
# "--moses-target-lang" ,"en"
# "--scoring" ,"sacrebleu",
# "--knnmt",
# "--quiet",
# "--k" ,"1",
# "--probe" ,"32",
# "--no-progress-bar",
# "--indexfile","/index_only.4096.index.vw.0"
# "--knn"-keytype ,"last_ffn_input",
# "--knn-embed-dim ,"512",
# "--no-load-keys"
# "--knn"-temp ,"10",
# "--knn"-sim-func ,"do_not_recomp_l2",
# "--skip-invalid-size-inputs-valid-test",
# "--lmbda","0.8",
# "--use-faiss-only" 


# 1000 sent pretrained labse all missmatch averaging 1000 sents 29706
# non pretained all missmatch 1000 sents 29714
# pretrained distluse averaging 400 sents 11776
# non pretrained 400 sents 11778
# pretrained distluse full 400 sents 11783
# labse full pretrained 400 sents 11782
# zeros like no pretrained 11778
# random 11729


# pretrained 
# missmatch number = 
# 11428
# number of correct 
# 1818


# no pretraining
# missmatch number = 
# 11439
# number of correct 
# 1813