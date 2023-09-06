import logging
import math
import os
import sys
import time
import sentencepiece as spm
import fileinput
import copy
from collections import namedtuple
import numpy as np
import pickle

import torch
import faiss

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.logging import progress_bar

from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders
from fairseq_cli.generate import main as generate_main

class KNNRetrieval ():
    def __init__(self,use_KNN=True):
        parser = options.get_generation_parser()
        self.args = options.parse_args_and_arch(parser)
        self.args.use_KNN=use_KNN

        # self.args.path="/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/training_dir/checkpoints_deen_pretrained/checkpoint_best.pt"
        # self.args.path="/mnt/mainz01eus/ababouelenin/projects/XY_CT/chkpt_50/checkpoint_best.pt"
        self.args.sentencepiece_model="/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/KNN-MT/vocab/spm.model"
        self.args.data = "/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/KNN-MT/vocab"
        # self.args.source_lang="de"
        # self.args.target_lang="en"

        # self.args.path="/mnt/mtcairogroup/Users/mohamedmaher/CTv2/KNN/fairseq_training/training/train_en-cs_lr5-4_WU4000_MT4096/checkpoint65.pt"
        # self.args.sentencepiece_model="/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/data/pretrain_data_enu_csy/mono/sentencepiece.bpe.model"
        # self.args.data = "/mnt/mtcairogroup/Users/mohamedmaher/CTv2/customer_bugs/DeMT-Data/en-cs-medpharma/train/dev/encoded_2/bin/"
        # self.args.source_lang="en"
        # self.args.target_lang="cs"

        self.args.moses_source_lang=self.args.source_lang
        self.args.moses_target_lang=self.args.target_lang
        self.args.model_overrides= "{'knn_keytype':'last_ffn_input'}"
        self.args.save_knns_csv="/home/ababouelenin/CustomTranslation/KNN-MT/save_knns_detailed/1m_datastore_k_1_beam_1"
        self.args.score_reference=True
        self.args.use_cuda=torch.cuda.is_available() and not self.args.cpu
        self.model=self.load_model()




    def load_model(self):
        logger = logging.getLogger('demo.load_model')

        utils.import_user_module(self.args)

        if self.args.max_tokens is None and self.args.max_sentences is None:
            self.args.max_tokens = 12000
        
        logger.info(self.args)
        task = tasks.setup_task(self.args)
        logger.info('loading model(s) from {}'.format(self.args.path))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.args.path),
            arg_overrides=eval(self.args.model_overrides),
            task=task,
            suffix=getattr(self.args, "checkpoint_suffix", ""),
        )
        for model in models:
            model.prepare_for_inference_(self.args)
            if self.args.fp16:
                model.half()
            if self.args.use_cuda:
                model.cuda()
        self.models = models

    def translate(self,sents,dsstore,use_KNN=False,interactive_bsz=4,source_lang="de",target_lang="en",english_centric=False,target_sents=None):
        self.args.sents =sents
        self.args.target_sents=target_sents
        self.args.dsstore= dsstore
        self.args.use_KNN=use_KNN
        self.args.interactive_bsz=interactive_bsz
        # self.args.source_lang
        self.args.source_lang=source_lang
        self.args.target_lang=target_lang
        self.args.english_centric=english_centric
        self.trans_results=copy.deepcopy(self.args)
        generate_main(self.trans_results,self.models)




        


def main():
    knn = KNNRetrieval()
    test_sent=["Fahrrad-Lift"]
    dsstore=dict()
    dsstore["Fahrrad-Lift"]=[{"source": "Fahrrad-Seitenstaender", "target": "Bike side stands", "score": 0.4545454545454546}, {"source": "Fahrrad mit auf Reisen.", "target": 
    "is no problem at all.","score":0.54455}]
    knn.translate(test_sent,dsstore)

if __name__ == "__main__":
    main()

