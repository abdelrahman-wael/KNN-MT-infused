TEMP=5
# SAVE_DIR=/mnt/mainz01eus/ababouelenin/projects/wmt2022/dataloader_investigation/fairseq_main_temp_5/multi_node_12enc_6dec_wmt_max_src_1024_16_1_continue_reset_dataloader
DATA_PATH=/mnt/mainz01eus/ababouelenin/projects/XY_CT/data/fairseq_preprocessed/all
# DATA_PATH=/mnt/mainz01eus/ababouelenin/projects/XY_CT/data/fairseq_preprocessed/all
LANG_PAIRS="de-en,en-de,fr-en,en-fr"
LANG_LIST="/home/aiscuser/KNN-MT/lang_list"
SAVE_DIR=/home/aiscuser/checkpoint_XY

# SAVE_DIR=/mnt/stander_main_storage/v-aaboueleni/projects/TemperatureSampling/checkpoint/subGermanic/${TAG}
# DATA_PATH=/mnt/stander_main_storage/v-aaboueleni/projects/TemperatureSampling/multiuat/data-bin/ted_8_diverse
# LANG_PAIRS="bos-eng,mar-eng,hin-eng,mkd-eng,ell-eng,bul-eng,fra-eng,kor-eng"



echo ${DATA_PATH}
echo ${LANG_PAIRS}
echo ${TEMP}

echo ${SAVE_DIR}

mkdir -p ${SAVE_DIR}/tensorboard_train
mkdir -p ${SAVE_DIR}/tensorboard_valid
mkdir -p ${SAVE_DIR}/tensorboard

python -m torch.distributed.launch --nproc_per_node=4 \
    --nnodes=1 --node_rank=0 --master_addr=10.8.32.230 \
    --master_port=12345 \
    $(which fairseq-train) ${DATA_PATH} \
    --encoder-normalize-before --decoder-normalize-before \
    --save-interval-updates 2056 \
    --layernorm-embedding \
    --task translation_multi_simple_epoch \
    --nprocs-per-node 4 \
    --sampling-method "temperature" \
    --lang-dict $LANG_LIST  \
    --sampling-temperature 5 \
    --skip-invalid-size-inputs-valid-test \
    --log-format tqdm --log-interval 100 \
    --arch transformer \
    --validate-interval-updates 512 \
    --share-decoder-input-output-embed \
    --max-epoch 300 \
    --lang-pairs ${LANG_PAIRS} \
    --attention-dropout 0 --relu-dropout 0 --weight-decay 0 \
    --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
    --warmup-init-lr 0 --warmup-updates 5000 --lr 0.0002121 \
    --decoder-embed-dim 512 \
    --encoder-embed-dim 512 \
    --encoder-layers 12 \
    --decoder-layers 6 \
    --encoder-ffn-embed-dim 2048 \
    --decoder-ffn-embed-dim 2048 \
    --dropout 0.1 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2034   --update-freq 16 \
    --encoder-normalize-before --decoder-normalize-before \
    --decoder-langtok \
    --fp16 \
    --tensorboard-logdir ${SAVE_DIR}/tensorboard \
    --save-dir ${SAVE_DIR} |& tee ${SAVE_DIR}/train-temp5.sh.${TEMP}.log
    