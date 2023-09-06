# DATA_STORES=/mnt/mainz01eus/v-enarouz/data/ELASTICSEARCH/de_en/data_store_de_en_2500_test_elastic_search_and_edit_distance_top_128_32
# OUTPUT_PATH=/home/ababouelenin/CustomTranslation/sequential_KNNMT/bash_script

SPM_VOCAB=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/vocab/deen_spm/sentencepiece.bpe.vocab
SPM_MODEL=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/vocab/deen_spm/sentencepiece.bpe.model
JOINT_DICT=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/vocab/deen_spm/dict.src.txt
JOINT_DICT_LOCAL=/home/mohamedmaher/JOINT_DICT
mkdir -p $JOINT_DICT_LOCAL
JOINT_DICT_LOCAL=$JOINT_DICT_LOCAL/dict.src.txt
cp $JOINT_DICT $JOINT_DICT_LOCAL
JOINT_DICT=$JOINT_DICT_LOCAL

DATA_STORES=/mnt/mainz01eus/v-enarouz/data/ELASTICSEARCH/de_en/data_store_de_en_2500_test_elastic_search_and_edit_distance_top_128_32/
TEST_SRC_PATH=/home/mohamedmaher/document_to_translate.de
OUTPUT_PATH=/home/mohamedmaher/translated_documnet


K=2 # K=2 
T=100
BEAM=5

MODEL_PATH=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/training_dir/checkpoints_deen_pretrained/checkpoint_best.pt
LOG_PATH_OUT=${OUTPUT_PATH}/logs
# TEST_TGT_PATH=/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/data/CT_vw/wv_ct_data/vw/ende/test/test.en-de.en

LOCAL_DS=/home/mohamedmaher/ds
PATH_TO_CHUNK_TEST=/home/mohamedmaher/de_chunks
SRC=de
TGT=en

dstore_training_data_folder=$LOCAL_DS
test_data_folder=$PATH_TO_CHUNK_TEST

rm -r $PATH_TO_CHUNK_TEST
mkdir $PATH_TO_CHUNK_TEST
echo "splitting testsets..."
split -a 5 -l 1 --numeric-suffixes=0  --additional-suffix=.$SRC $TEST_SRC_PATH $PATH_TO_CHUNK_TEST/sent_


# rm -r $LOCAL_DS
# python /mnt/mtcairogroup/Users/mohamedmaher/code/parse_elastic_search_json_to_ds.py \
# --es-output $DATA_STORES \
# --output $LOCAL_DS

i=0
# # to=2499
to=$(cat $TEST_SRC_PATH | wc -l )
# to=$(( to - 1 ))
echo $i
echo $to
while [ "$i" -le $to ]; do 

    echo "starting datastore_$i ..."

    SENT_NAME=$(echo 00000$i | tail -c 6)
    
    FOLDER_PATH=$LOCAL_DS/test_sample_$i.ds
    SPM_ENCODE_OUTPUT_DIR=$FOLDER_PATH/encoded/spm
    final_output=$FOLDER_PATH/encoded/bin/
    DATA_STORE=$FOLDER_PATH/dstore
    DATA_PATH=$final_output
    BIN_FILE=train.de-en.en.bin
    LOG_PATH=$FOLDER_PATH/logs

    # rm -f -r $SPM_ENCODE_OUTPUT_DIR
    # mkdir -p $SPM_ENCODE_OUTPUT_DIR
    rm -f -r $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin 
    mkdir $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin 
    # rm -f -r  $final_output
    # mkdir -p $final_output
    # rm -f -r $DATA_STORE
    # mkdir $DATA_STORE
    # rm -f -r $LOG_PATH
    # mkdir $LOG_PATH

    echo "spm datastore_$i ..."
    start=`date +%s.%N`
    # spm_encode \
    #     --model $SPM_MODEL \
    #     --output_format=piece \
    #     --input $FOLDER_PATH/train.de-en.de \
    #     --output $SPM_ENCODE_OUTPUT_DIR/train.de-en.de > $LOG_PATH/spm_datastore_$i
    # spm_encode \
    #     --model $SPM_MODEL \
    #     --output_format=piece \
    #     --input $FOLDER_PATH/train.de-en.en \
    #     --output $SPM_ENCODE_OUTPUT_DIR/train.de-en.en > $LOG_PATH/spm_datastore_$i
    spm_encode \
        --model $SPM_MODEL \
        --output_format=piece \
        --input $PATH_TO_CHUNK_TEST/sent_$SENT_NAME.$SRC \
        --output $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.$SRC 
    end=`date +%s.%N`
    echo $( echo "$end - $start" | bc -l )
    echo "preprocess datastore_$i ..."
    start=`date +%s.%N`
    # python ~/KNN-MT/fairseq_cli/preprocess.py --source-lang $SRC --target-lang $TGT \
    #     --trainpref $SPM_ENCODE_OUTPUT_DIR/train.de-en \
    #     --destdir $final_output \
    #     --srcdict $JOINT_DICT \
    #     --tgtdict $JOINT_DICT \
    #     --workers 50 > $LOG_PATH/preprocess_datastore_$i
    
    
    python ~/KNN-MT/fairseq_cli/preprocess.py --source-lang $SRC --only-source \
        --trainpref $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME \
        --destdir $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin \
        --srcdict $JOINT_DICT \
        --tgtdict $JOINT_DICT \
        --workers 50 > $LOG_PATH/preprocess_datastore_$i
        
    cp $JOINT_DICT $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin/dict.en.txt
    cp $JOINT_DICT $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin/dict.de.txt
    mv $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin/train.de-None.de.bin $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin/train.de-en.de.bin
    mv $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin/train.de-None.de.idx $PATH_TO_CHUNK_TEST/sent_spm_$SENT_NAME.bin/train.de-en.de.idx
    #################
    end=`date +%s.%N`
    echo $( echo "$end - $start" | bc -l )
    i=$(( i + 1))
    
done


mkdir -p $OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fairseq_cli/generate.py /home/mohamedmaher/dump \
    --adaptive-temperature $T \
    --test-data-folder $test_data_folder \
    --dstore-training-data-folder $dstore_training_data_folder \
    --max-tokens 4096 \
    --save-knn-dstore \
    --gen-subset train \
    --knn-add-to-idx \
    --path $MODEL_PATH \
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
    --max-tokens 4096 \
    --test-count 1 \
    --use-faiss-only > $LOG_PATH_OUT

echo $LOG_PATH_OUT
grep D-0 $LOG_PATH_OUT | awk '{$1=$2=""; print $0}' > $OUTPUT_PATH/pred
# grep D-0 $LOG_PATH_OUT | awk '{$1=$2=""; print $0}' | sacrebleu $TEST_TGT_PATH > $OUTPUT_PATH/scores.txt
# grep D-0 $LOG_PATH_OUT | awk '{$1=$2=""; print $0}' | sacrebleu $TEST_TGT_PATH