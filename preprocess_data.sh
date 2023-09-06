SPM_MODEL=/mnt/mainz01eus/ababouelenin/projects/XY_CT/data/raw/mono/spm.model
RAW_DATA=/mnt/mainz01eus/ababouelenin/projects/XY_CT/data/dev/raw
DIRECTION=en-de
SRC=en
TGT=de
SPM_ENCODE_OUTPUT_DIR=/mnt/mainz01eus/ababouelenin/projects/XY_CT/data/dev/spm_encoded
JOINT_DICT=/mnt/mainz01eus/ababouelenin/projects/XY_CT/data/raw/mono/dict.src.txt
FAIRSEQ_PREPROCESS_OUTPUT_DIR=/mnt/mainz01eus/ababouelenin/projects/XY_CT/data/dev/fairseq_preprocessed

mkdir -p $SPM_ENCODE_OUTPUT_DIR/$SRC$TGT
mkdir -p $FAIRSEQ_PREPROCESS_OUTPUT_DIR/$SRC$TGT


spm_encode \
    --model $SPM_MODEL \
    --output_format=piece \
    --input $RAW_DATA/$SRC$TGT/dev.$DIRECTION.$SRC \
    --output $SPM_ENCODE_OUTPUT_DIR/$SRC$TGT/dev.bpe.$DIRECTION.$SRC

spm_encode \
    --model $SPM_MODEL \
    --output_format=piece \
    --input $RAW_DATA/$SRC$TGT/dev.$DIRECTION.$TGT \
    --output $SPM_ENCODE_OUTPUT_DIR/$SRC$TGT/dev.bpe.$DIRECTION.$TGT


fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --validpref $SPM_ENCODE_OUTPUT_DIR/$SRC$TGT/dev.bpe.$DIRECTION \
    --destdir $FAIRSEQ_PREPROCESS_OUTPUT_DIR/$SRC$TGT \
    --srcdict $JOINT_DICT \
    --tgtdict $JOINT_DICT 
    # --workers 50