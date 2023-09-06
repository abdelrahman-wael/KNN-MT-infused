#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import logging
import math
import os
import sys
import time
import sentencepiece as spm
import fileinput
import sentencepiece as spm
# from transformers import NllbTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

from collections import namedtuple
import numpy as np
import pickle

import torch
import faiss

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.logging import progress_bar

from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders
# import pandas as pd

from tqdm import tqdm

Batch = namedtuple('Batch', 'ids src_tokens src_lengths target target_tokens')



# def buffered_read(src , tgt , buffer_size):
#     buffer = []
#     with fileinput.input(files=[src], openhook=fileinput.hook_encoded("utf-8")) as h:
#         with fileinput.input(files=[tgt], openhook=fileinput.hook_encoded("utf-8")) as h:
#             for src_str in h:
#                 buffer.append(src_str.strip())
#                 if len(buffer) >= buffer_size:
#                     yield buffer
#                     buffer = []

#     if len(buffer) > 0:
#         yield buffer
def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])


def make_batches(src_lines,tgt_lines, args, task, encode_fn ,models,
                    append_eos=True,
                    replaced_consumer=replaced_consumer,
                    reverse_order=False,
                    offset=0,):

    # encode_line(
    #                     line=line,
    #                     line_tokenizer=tokenize,
    #                     add_if_not_exist=False,
    #                     consumer=replaced_consumer,
    #                     append_eos=append_eos,
    #                     reverse_order=reverse_order,
    #                 )
    # import pdb;pdb.set_trace()
    src_tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str.strip().strip("\ufeff")),append_eos=True, add_if_not_exist=False
        ).long()
        for src_str in src_lines
    ]
    # import pdb;pdb.set_trace()
    src_lengths = [t.numel() for t in src_tokens]

    if tgt_lines:
        tgt_tokens =[
            task.source_dictionary.encode_line(
                encode_fn(tgt_str.strip().strip("\ufeff")), add_if_not_exist=False,append_eos=True
            ).long()
            for tgt_str in tgt_lines
        ]

        tgt_lengths = [t.numel() for t in tgt_tokens]
        return task.get_batch_iterator(
                                        dataset=task.build_dataset_for_inference(src_tokens=src_tokens,
                                                                                src_lengths=src_lengths,
                                        # src_sizes=src_lengths,
                                                                                tgt_tokens=tgt_tokens,
                                                                                tgt_lengths=tgt_lengths)
                                                                                # dic=task.source_dictionary)
                                                                                ,
                                        max_tokens=args.max_tokens,
                                        max_sentences=args.max_sentences,
                                        max_positions=utils.resolve_max_positions(
                                        task.max_positions(),
                                        *[model.max_positions() for model in models]
                                        ),
                                        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                                        required_batch_size_multiple=args.required_batch_size_multiple,
                                        num_shards=args.num_shards,
                                        shard_id=args.shard_id,
                                        num_workers=args.num_workers,
                                        ).next_epoch_itr(shuffle=False)
    # itr = task.get_batch_iterator(
    #     dataset=task.build_dataset_for_inference(src_tokens=src_tokens,src_sizes=src_lengths,
    #                                             tgt_tokens=tgt_tokens,tgt_sizes=tgt_lengths,
    #                                             dic=task.source_dictionary),
    #     max_tokens=args.max_tokens,
    #     max_sentences=args.max_sentences,
    #     max_positions=max_positions,
    #     ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    # ).next_epoch_itr(shuffle=False)

    return task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(src_tokens=src_tokens,src_lengths=src_lengths),
                                                
                                                # dic=task.source_dictionary),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

    # for batch in itr:
    #     yield Batch(
    #         ids=batch['id'],
    #         src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],target=[],target_tokens=[]
    #     )


def read_files(src_file,tgt_file=None):
            src_file_pointer = open(src_file,"r",encoding="utf-8")
            src_lines = src_file_pointer.readlines()
            if tgt_file:
                tgt_file_pointer=open(tgt_file,"r",encoding="utf-8")
                tgt_lines = tgt_file_pointer.readlines()
                return src_lines , tgt_lines
            return src_lines

def main(args,models=None):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.dataset_impl == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w', buffering=1, encoding='utf-8') as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout , models=models)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, 'symbols_to_strip_from_output'):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}

@torch.no_grad()
def _main(args, output_file,models=None):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.generate')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    
    # logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    results=[]
    # Load dataset splits
    # args.score_reference=True
    task = tasks.setup_task(args)
    logger.info('loading model(s) from {}'.format(args.path))
    
    # models, _model_args = checkpoint_utils.load_model_ensemble(
    #     utils.split_paths(args.path),
    #     arg_overrides=eval(args.model_overrides),
    #     task=task,
    #     suffix=getattr(args, "checkpoint_suffix", ""),
    # )
    
    # for model in models:
    #     model.prepare_for_inference_(args)
    #     if args.fp16:
    #         model.half()
    #     if use_cuda:
    #         model.cuda()
    # test_count = len(os.listdir(args.test_data_folder))
    model=models[0]
    # import pdb;pdb.set_trace()
    st = time.time()
    args.is_batch = True
    args.ds_hypo=None
    batch_src_ds = []
    batch_tgt_ds = []
    batch_src_sent = []
    batch_tgt_sent = []
    batched = 0
    args.knns_missmatch = None
    counter = 0
    if args.use_KNN == False:
        index=None
    else:
        index=faiss.index_factory(args.knn_embed_dim, 'IDMap,Flat', faiss.METRIC_L2)
        # res=faiss.StandardGpuResources()
        # index=faiss.index_cpu_to_gpu(res, 0, index)
    # args.total_decoding_ds_time=0
    args.english_centric=False
    args.use_pretrained=True
    args.ds_hypos=[]
    tgt_dict = task.target_dictionary
    if args.english_centric:
            args.final_target_lang=args.target_lang
            args.initial_source_lang=args.source_lang
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    sp = spm.SentencePieceProcessor(args.sentencepiece_model)
    args.sp = sp
    args.special_char = args.sp.encode_as_pieces("help")[0][-1]
    args.correct_knns=None
    if args.use_pretrained :
        # args.pretrained_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        args.pretrained_model = SentenceTransformer('sentence-transformers/LaBSE')
        # args.pretrained_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    
    args.sp_model=sp
    for test_sent in tqdm(args.sents):
    # for sent_id in range(test_count//3):
        # import pdb
        # pdb.set_trace()
    # if args.use_KNN:
        
        # args.data=args.dstore_training_data
        # task.load_dataset(args.gen_subset)

        # Set dictionaries

        # Load ensemble

        # Optimize ensemble for generation

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(args.replace_unk)

        # task.build_dataset_for_inference()
        # Load dataset (possibly sharded)
        tokenizer = None
        bpe = encoders.build_bpe(args)
        

        def encode_fn(x):
            # print("before encoding ",x)
            if tokenizer is not None:
                x = tokenizer.encode(x)
            if bpe is not None:
                x = bpe.encode(x)
            # print("after encoding ",x)
            return x

        # src_file = os.path.join(args.dstore_training_data_folder,"test_sample_"+str(sent_id)+".ds","train.de-en.de")
        # tgt_file = os.path.join(args.dstore_training_data_folder,"test_sample_"+str(sent_id)+".ds","train.de-en.en")
        if args.use_KNN:
            parallel_ds=args.dsstore[test_sent]
        
        # src_lines,tgt_lines = read_files(src_file,tgt_file)
        
        if args.is_batch == True:
            if args.use_KNN:
                src_lines = [ ds_sent["source"] for ds_sent in parallel_ds ]
                tgt_lines = [ ds_sent["target"] for ds_sent in parallel_ds ]

                batch_src_ds = batch_src_ds + src_lines
                batch_tgt_ds = batch_tgt_ds + tgt_lines

            batch_src_sent = batch_src_sent + [test_sent]
            batch_tgt_sent = batch_tgt_sent + [args.target_sents[counter]]

            batched += 1
            counter += 1

            if batched < args.interactive_bsz:
                if len(args.sents) != counter:
                    continue


        else:
            batch_src_sent = [test_sent]
            batch_src_ds = [ ds_sent["source"] for ds_sent in parallel_ds ]
            batch_tgt_ds = [ ds_sent["target"] for ds_sent in parallel_ds ]


        # start of datastore creation
        def decode_fn(x):
                if bpe is not None:
                    x = bpe.decode(x)
                if tokenizer is not None:
                    x = tokenizer.decode(x)
                return x

        def decode_fn(x):
                if bpe is not None:
                    x = bpe.decode(x)
                if tokenizer is not None:
                    x = tokenizer.decode(x)
                return x
        wps_meter = TimeMeter()  

        def translate(src_lang,tgt_lang,use_KNN=False,index=None,translate_ds=False):
            num_sentences = 0
            translation_output=[]
            args.target_lang=tgt_lang
            args.source_lang=src_lang
            args.gen_subset = "train"
            args.knnmt=use_KNN
            args.knn_add_to_idx=False
            args.score_reference=None
            args.no_load_keys=True
            args.knn_add_to_idx=False
            args.save_knn_dstore=False
            task = tasks.setup_task(args)
            generator = task.build_generator(models, args)
            if args.use_KNN == False:
                index = None
            if not translate_ds:
                itr = make_batches(args=args,src_lines=batch_src_sent,
                                tgt_lines=batch_tgt_sent
                                ,models=models ,
                                task=task ,  encode_fn=encode_fn)
            else:
                itr = make_batches(args=args,src_lines=batch_src_ds,
                                tgt_lines=batch_tgt_sent,models=models ,
                                task=task ,  encode_fn=encode_fn)

            
            progress = progress_bar.progress_bar(
                itr,
                log_format=args.log_format,
                log_interval=args.log_interval,
                default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
            )

            
            for idx, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if 'net_input' not in sample:
                    continue

                ## For processing in parallel
                if args.save_knn_dstore and to_skip > 0:
                    num_samples = sample['target'].shape[0]
                    if to_skip - num_samples > 0:
                        to_skip -= num_samples
                        target_tokens = utils.strip_pad(sample['target'], tgt_dict.pad()).int().cpu()
                        start_pos += len(target_tokens)
                        continue

                    for i, sample_id in enumerate(sample['id'].tolist()):
                        if to_skip > 0:
                            to_skip -= 1
                            target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                            start_pos += len(target_tokens)
                        else:
                            tgt_tokens = utils.strip_pad(sample['target'][i:], tgt_dict.pad()).int().cpu()
                            new_sample = {
                                    'id': sample['id'][i:],
                                    'nsentences': len(sample['id'][i:]),
                                    'ntokens': len(tgt_tokens),
                                    'net_input': {
                                        'src_tokens': sample['net_input']['src_tokens'][i:],
                                        'src_lengths': sample['net_input']['src_lengths'][i:],
                                        'prev_output_tokens': sample['net_input']['prev_output_tokens'][i:],
                                    },
                                    'target': sample['target'][i:]

                            }
                            sample = new_sample
                            break

                    print('Starting the saving at location %d in the mmap' % start_pos)
                ## For processing in parallel

                prefix_tokens = None
                if args.prefix_size > 0:
                    prefix_tokens = sample['target'][:, :args.prefix_size]
                
                hypos = task.inference_step(generator, models, sample, prefix_tokens,index=index)
                

                if args.save_knns_csv and args.save_knns:
                    src_sent_decoded = sp.decode([task.tgt_dict[token] for token in sample["net_input"]["src_tokens"][0]])
                    tgt_sent_decoded = sp.decode([task.tgt_dict[token] for token in sample["target"][0]])
                    knns_decoded = sp.decode([task.tgt_dict[int(token)] for token in hypos[0][0]["knns"]])
                    knn_dists = [dist for dist in hypos[0][0]["dists"]]
                    knns_tokens = [sp.decode(task.tgt_dict[int(token)]) for token in hypos[0][0]["knns"]]
                    pd_dic={"src_decoded":[src_sent_decoded,0],"tgt_decoded":[tgt_sent_decoded,0],"knns_decoded":[knns_decoded,0]}
                    pd_dic.update({"token_"+str(i):[token,round(float(dist),4)] for i,(dist,token) in zip(range(len(knn_dists)),zip(knn_dists,knns_tokens))})
                    sentence_id=str(sample["id"].item())
                    # pd.DataFrame(pd_dic).to_csv(args.save_knns_csv+"/sentence_"+sentence_id+".csv")
                num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
                #print(hypos[0][0]['tokens'])
                #exit(0)
                # gen_timer.stop(num_generated_tokens)

                if args.knn_add_to_idx:
                    saving = sample['ntokens']
                    if args.drop_lang_tok:
                        saving = sample['ntokens'] - sample['target'].shape[0]
                    keys = np.zeros([saving, model.decoder.embed_dim], dtype=np.float32)
                    addids = np.zeros([saving], dtype=np.int)
                    save_idx = 0

                for i, sample_id in enumerate(sample['id'].tolist()):
                    # loop_start = time.time()
                    has_target = sample['target'] is not None
                    #print(sample['target'][i])

                    # Remove padding
                    if 'src_tokens' in sample['net_input']:
                        src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                    else:
                        src_tokens = None

                    target_tokens = None
                    if has_target:
                        target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                    if args.score_reference:
                        continue

                    # Either retrieve the original sentences or regenerate them from tokens.
                    if align_dict is not None:
                        src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                        target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                    else:
                        if src_dict is not None:
                            src_str = src_dict.string(src_tokens, args.remove_bpe)
                        else:
                            src_str = ""
                        #print(get_symbols_to_strip_from_output(generator))
                        if has_target:
                            target_str = tgt_dict.string(
                                target_tokens,
                                args.remove_bpe,
                                escape_unk=True,
                                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                            )

                    src_str = decode_fn(src_str)
                    if has_target:
                        target_str = decode_fn(target_str)

                    if not args.quiet:
                        if src_dict is not None:
                            print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                        if has_target:
                            print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

                    # Process top predictions
                    for j, hypo in enumerate(hypos[i][:args.nbest]):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'],
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                        )
                        sp=spm.SentencePieceProcessor(args.sentencepiece_model)
                        detok_hypo_str = sp.decode(hypo_str.split(" "))
                        detok_hypo_str=detok_hypo_str.replace("__en__","")
                        detok_hypo_str=detok_hypo_str.replace("__de__","")
                        detok_hypo_str=detok_hypo_str.replace("__fr__","")
                        translation_output.append(detok_hypo_str)
                        if not args.quiet:
                        # if True:
                            score = hypo['score'] / math.log(2)  # convert to base 2
                            # original hypothesis (after tokenization and BPE)
                            print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                            # detokenized hypothesis
                            print('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                            print('P-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(
                                    lambda x: '{:.4f}'.format(x),
                                    # convert from base e to base 2
                                    hypo['positional_scores'].div_(math.log(2)).tolist(),
                                ))
                            ), file=output_file)

                            if args.print_alignment:
                                print('A-{}\t{}'.format(
                                    sample_id,
                                    ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                                ), file=output_file)

                            if args.print_step:
                                print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                            if getattr(args, 'retain_iter_history', False):
                                for step, h in enumerate(hypo['history']):
                                    _, h_str, _ = utils.post_process_prediction(
                                        hypo_tokens=h['tokens'].int().cpu(),
                                        src_str=src_str,
                                        alignment=None,
                                        align_dict=None,
                                        tgt_dict=tgt_dict,
                                        remove_bpe=None,
                                    )
                                    print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)
                        # import pdb;pdb.set_trace()
                        # Score only the top hypothesis
                        if has_target and j == 0:
                            if align_dict is not None or args.remove_bpe is not None:
                                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                                target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                                hypo_tokens = tgt_dict.encode_line(detok_hypo_str, add_if_not_exist=True)
                            if hasattr(scorer, 'add_string'):
                                scorer.add_string(target_str, detok_hypo_str)
                            else:
                                scorer.add(target_tokens, hypo_tokens)


                    if args.knn_start > -1 and knn_num_samples_proc == args.knn_proc:
                        break
                    if args.save_knn_subset and total_saved >= args.save_knn_subset_num:
                        break
                    #if i > 10:
                    #    break
                if args.knn_start > -1 and knn_num_samples_proc == args.knn_proc:
                    break
                if args.save_knn_subset and total_saved >= args.save_knn_subset_num:
                    break

                wps_meter.update(num_generated_tokens)
                progress.log({'wps': round(wps_meter.avg)})
                num_sentences += sample["nsentences"] if "nsentences" in sample else sample['id'].numel()
            return translation_output
        # import pdb;pdb.set_trace()
        

        if args.english_centric:
            batch_src_sent=translate(src_lang=args.initial_source_lang,tgt_lang="en") 


        # if args.english_centric:


        if args.use_KNN:
            if args.english_centric:
                batch_src_ds = translate(src_lang=args.initial_source_lang,tgt_lang="en",translate_ds=True)
            args.score_reference=True
            args.knn_add_to_idx=True
            args.knn_add_to_idx=True
            args.save_knn_dstore=True
            args.knnmt = False
            adding_to_faiss=0
            task = tasks.setup_task(args)
            
            # decoding_ds_time=time.time()
            itr = make_batches(args=args,src_lines=batch_src_ds,
                            tgt_lines=batch_tgt_ds,models=models,
                            task=task ,  encode_fn=encode_fn)
            # args.total_decoding_ds_time+=time.time()-decoding_ds_time
            # print(args.total_decoding_ds_time)
            # itr = task.get_batch_iterator(
            #     dataset=task.dataset(args.gen_subset),
            #     max_tokens=args.max_tokens,
            #     max_sentences=args.max_sentences,
            #     max_positions=utils.resolve_max_positions(
            #     task.max_positions(),
            #     *[model.max_positions() for model in models]
            #     ),
            #     ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            #     required_batch_size_multiple=args.required_batch_size_multiple,
            #     num_shards=args.num_shards,
            #     shard_id=args.shard_id,
            #     num_workers=args.num_workers,
            # ).next_epoch_itr(shuffle=False)
            
            # itr = task.get_batch_iterator(
            #     dataset=task.dataset(args.gen_subset),
            #     max_tokens=args.max_tokens,
            #     max_sentences=args.max_sentences,
            #     max_positions=utils.resolve_max_positions(
            #     task.max_positions(),
            #     *[model.max_positions() for model in models]
            #     ),
            #     ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            #     required_batch_size_multiple=args.required_batch_size_multiple,
            #     num_shards=args.num_shards,
            #     shard_id=args.shard_id,
            #     num_workers=args.num_workers,
            # ).next_epoch_itr(shuffle=False)
            progress = progress_bar.progress_bar(
                itr,
                log_format=args.log_format,
                log_interval=args.log_interval,
                default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
            )

            # Initialize generator
            # gen_timer = StopwatchMeter()
            generator = task.build_generator(models, args)

            # Handle tokenization and BPE
            

            def decode_fn(x):
                if bpe is not None:
                    x = bpe.decode(x)
                if tokenizer is not None:
                    x = tokenizer.decode(x)
                return x

            ## knn saving code
            if args.save_knn_dstore:
                # print('keytype being saved:', args.knn_keytype)
                if args.knn_start > -1:
                    chunk_size = 100000
                    if args.dstore_fp16:
                        # print('Saving fp16')
                        dstore_keys = np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float16)
                        dstore_vals = np.zeros([chunk_size, 1], dtype=np.int16)
                    else:
                        # print('Saving fp32')
                        dstore_keys = np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float32)
                        dstore_vals = np.zeros([chunk_size, 1], dtype=np.int)

                else:
                    assert not (args.save_knn_subset and args.knn_add_to_idx)
                    dstore_size = args.dstore_size
                    if args.save_knn_subset:
                        dstore_size = args.save_knn_subset_num
                    if args.dstore_fp16:
                        print('Saving fp16')
                        if args.knn_add_to_idx:
                            faiss_indices = []
                            for tindex in args.trained_index:
                                print("Reading trained index from %s" % tindex)
                                faiss_indices.append(faiss.read_index(tindex))

                                if args.knn_q2gpu:
                                    assert len(args.trained_index) == 1
                                    print("Moving quantizer to GPU")
                                    index_ivf = faiss.extract_index_ivf(faiss_indices[0])
                                    quantizer = index_ivf.quantizer
                                    quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer, ngpu=1)
                                    index_ivf.quantizer = quantizer_gpu
                        else:
                            dstore_keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+', shape=(dstore_size, model.decoder.embed_dim))
                            dstore_vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int16, mode='w+', shape=(dstore_size, 1))
                    else:
                        # print('Saving fp32')
                        if args.knn_add_to_idx:
                            faiss_indices = []
                            # index = faiss.index_factory(512, 'IDMap,Flat', faiss.METRIC_L2)
                            
                            # import pdb
                            # pdb.set_trace()
                            # faiss_indices.append(index)
                            index.reset()
                            # for tindex in args.trained_index:
                            #     print("Reading trained index from %s" % tindex)
                                


                                # if args.knn_q2gpu:
                                #     assert len(args.trained_index) == 1
                                #     print("Moving quantizer to GPU")
                                    
                                #     index = faiss_indices[0]
                                #     quantizer_gpu = faiss.index_cpu_to_all_gpus(index, ngpu=8)
                                #     index_ivf = quantizer_gpu
                    
                        else:
                            dstore_keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='w+', shape=(dstore_size, model.decoder.embed_dim))
                            dstore_vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='w+', shape=(dstore_size, 1))

                dstore_idx = 0
                total_saved = 0
                knn_num_samples_proc = 0
                to_skip = -1
                if args.knn_start > -1:
                    to_skip = args.knn_start # examples
                    start_pos = 0
                
                # save the sample ids and the lengths for backtracking the neighbors
                sample_order_lens = [[],[]]
            if args.knn_add_to_idx:
                    adding_to_faiss = 0
            if args.knnmt and args.save_knns:
                to_save_objects = []
            ## knn saving code
            
            # Generate and compute BLEU score
            scorer = scoring.scoring_utils.build_scorer(args, tgt_dict)
            # import pdb
            # pdb.set_trace()
            
            has_target = True
            
            start_time = time.time()

            # CREATING DATASTORE !!!!!!!! 

            # args.knnmt=False
            
            s_time_all = time.time()
            # index = faiss.index_factory(512, 'IDMap,Flat', faiss.METRIC_L2)
            # for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            #     src_tokens = batch.src_tokens
            #     src_lengths = batch.src_lengths
            #     if use_cuda:
            #         src_tokens = src_tokens.cuda()
            #         src_lengths = src_lengths.cuda()

            #     sample = {
            #         'net_input': {
            #             'src_tokens': src_tokens,
            #             'src_lengths': src_lengths,
            #         },

            #     }

            # for inputs in buffered_read(args.input, args.buffer_size):
            #     results = []
            #     for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            #         src_tokens = batch.src_tokens
            #         src_lengths = batch.src_lengths
            #         if use_cuda:
            #             src_tokens = src_tokens.cuda()
            #             src_lengths = src_lengths.cuda()

            #         sample = {
            #             'net_input': {
            #                 'src_tokens': src_tokens,
            #                 'src_lengths': src_lengths,
            #             },
            #         }


            for idx, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if 'net_input' not in sample:
                    continue

                # For processing in parallel
                if args.save_knn_dstore and to_skip > 0:
                    num_samples = sample['target'].shape[0]
                    if to_skip - num_samples > 0:
                        to_skip -= num_samples
                        target_tokens = utils.strip_pad(sample['target'], tgt_dict.pad()).int().cpu()
                        start_pos += len(target_tokens)
                        continue

                    for i, sample_id in enumerate(sample['id'].tolist()):
                        if to_skip > 0:
                            to_skip -= 1
                            target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                            start_pos += len(target_tokens)
                        else:
                            tgt_tokens = utils.strip_pad(sample['target'][i:], tgt_dict.pad()).int().cpu()
                            new_sample = {
                                    'id': sample['id'][i:],
                                    'nsentences': len(sample['id'][i:]),
                                    'ntokens': len(tgt_tokens),
                                    'net_input': {
                                        'src_tokens': sample['net_input']['src_tokens'][i:],
                                        'src_lengths': sample['net_input']['src_lengths'][i:],
                                        'prev_output_tokens': sample['net_input']['prev_output_tokens'][i:],
                                    },
                                    'target': sample['target'][i:]

                            }
                            sample = new_sample
                            break

                    # print('Starting the saving at location %d in the mmap' % start_pos)
                ## For processing in parallel

                prefix_tokens = None
                if args.prefix_size > 0:
                    prefix_tokens = sample['target'][:, :args.prefix_size]

                #print('target', sample['target'].shape)
                # gen_timer.start()
                args.ds_hypos=[]
                hypos = task.inference_step(generator, models, sample, prefix_tokens, index=None)
                args.ds_hypos.append(hypos)
                # print("**************" )
                # print(args.average_dist)
                #exit()
                # need to dump data here!!
                # sample["net_input"]["src_tokens"] shape = num_sent*tokens
                # if args.save_knns_csv and args.save_knns:
                #     src_sent_decoded = sp.decode([task.tgt_dict[token] for token in sample["net_input"]["src_tokens"][0]])
                #     tgt_sent_decoded = sp.decode([task.tgt_dict[token] for token in sample["target"][0]])
                #     knns_decoded = sp.decode([task.tgt_dict[int(token)] for token in hypos[0][0]["knns"]])
                #     knn_dists = [dist for dist in hypos[0][0]["dists"]]
                #     knns_tokens = [sp.decode(task.tgt_dict[int(token)]) for token in hypos[0][0]["knns"]]
                #     pd_dic={"src_decoded":[src_sent_decoded,0],"tgt_decoded":[tgt_sent_decoded,0],"knns_decoded":[knns_decoded,0]}
                #     pd_dic.update({"token_"+str(i):[token,round(float(dist),4)] for i,(dist,token) in zip(range(len(knn_dists)),zip(knn_dists,knns_tokens))})
                #     sentence_id=str(sample["id"].item())
                #     pd.DataFrame(pd_dic).to_csv(args.save_knns_csv+"/sentence_"+sentence_id+".csv")
                num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
                #print(hypos[0][0]['tokens'])
                #exit(0)
                # gen_timer.stop(num_generated_tokens)

                if args.knn_add_to_idx:
                    saving = sample['ntokens']
                    if args.drop_lang_tok:
                        saving = sample['ntokens'] - sample['target'].shape[0]
                    keys = np.zeros([saving, args.knn_embed_dim], dtype=np.float32)
                    addids = np.zeros([saving], dtype=np.int)
                    save_idx = 0

                for i, sample_id in enumerate(sample['id'].tolist()):
                    has_target = sample['target'] is not None
                    #print(sample['target'][i])

                    # Remove padding
                    if 'src_tokens' in sample['net_input']:
                        src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                    else:
                        src_tokens = None

                    target_tokens = None
                    if has_target:
                        target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                    # print(len(hypos))
                    # print(hypos[i][0]['tokens'].shape)
                    # print(len(target_tokens))
                    # print(hypos[i][0]['tokens'])
                    # knn saving code
                    if args.save_knn_dstore:
                        hypo = hypos[i][0]
                        num_items = len(hypo['tokens'])
                        #print(num_items, hypo['dstore_keys_mt'].shape)
                        #print(hypo['tokens'])
                        #print(hypo['dstore_keys_mt'])
                        #exit(0)
                        #sample_order_lens[0].append(sample_id)
                        #sample_order_lens[1].append(num_items)
                        #if dstore_idx + shape[0] > args.dstore_size:
                        #    shape = [args.dstore_size - dstore_idx]
                        #    hypo['dstore_keys_mt'] = hypo['dstore_keys_mt'][:shape[0]]
                        if args.knn_start > -1:
                            if dstore_idx + num_items > dstore_keys.shape[0]:
                                if args.dstore_fp16:
                                    dstore_keys = np.concatenate([dstore_keys, np.zeros([chunk_size, args.knn_embed_dim], dtype=np.float16)], axis=0)
                                    dstore_vals = np.concatenate([dstore_vals, np.zeros([chunk_size, 1], dtype=np.int16)], axis=0)
                                else:
                                    dstore_keys = np.concatenate([dstore_keys, np.zeros([chunk_size, args.knn_embed_dim], dtype=np.float32)], axis=0)
                                    dstore_vals = np.concatenate([dstore_vals, np.zeros([chunk_size, 1], dtype=np.int)], axis=0)

                        skip = 0
                        if args.drop_lang_tok:
                            skip += 1

                        if args.save_knn_subset:
                            if total_saved + num_items - skip > args.save_knn_subset_num:
                                num_items = args.save_knn_subset_num - total_saved + skip

                        if args.knn_add_to_idx:
                            keys[save_idx:save_idx+num_items-skip] = hypo['dstore_keys_mt'][skip:num_items].view(
                                    -1, args.knn_embed_dim).cpu().numpy().astype(np.float32)
                            addids[save_idx:save_idx+num_items-skip] = hypo['tokens'][skip:num_items].view(
                                    -1).cpu().numpy().astype(np.int)
                            save_idx += num_items - skip

                        if not args.knn_add_to_idx:
                            if args.dstore_fp16:
                                dstore_keys[dstore_idx:num_items-skip+dstore_idx] = hypo['dstore_keys_mt'][skip:num_items].view(
                                        -1, args.knn_embed_dim).cpu().numpy().astype(np.float16)
                                dstore_vals[dstore_idx:num_items-skip+dstore_idx] = hypo['tokens'][skip:num_items].view(
                                        -1, 1).cpu().numpy().astype(np.int16)
                            else:
                                dstore_keys[dstore_idx:num_items-skip+dstore_idx] = hypo['dstore_keys_mt'][skip:num_items].view(
                                        -1, args.knn_embed_dim).cpu().numpy().astype(np.float32)
                                dstore_vals[dstore_idx:num_items-skip+dstore_idx] = hypo['tokens'][skip:num_items].view(
                                        -1, 1).cpu().numpy().astype(np.int)

                        dstore_idx += num_items - skip
                        total_saved += num_items - skip
                        knn_num_samples_proc += 1
                    ## knn saving code
                    if args.score_reference:
                        continue

                    # error analysis knnmt: save knns, vals and probs
                    if args.knnmt and args.save_knns:
                        to_save_objects.append(
                                {
                                    "id": sample_id,
                                    "src": src_tokens,
                                    "tgt": target_tokens,
                                    "hypo": hypos[i],
                                }
                            )
                    ## error analysis knnmt: save knns, vals and probs

                    # Either retrieve the original sentences or regenerate them from tokens.
                    if align_dict is not None:
                        src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                        target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                    else:
                        if src_dict is not None:
                            src_str = src_dict.string(src_tokens, args.remove_bpe)
                        else:
                            src_str = ""
                        #print(get_symbols_to_strip_from_output(generator))
                        if has_target:
                            target_str = tgt_dict.string(
                                target_tokens,
                                args.remove_bpe,
                                escape_unk=True,
                                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                            )

                    src_str = decode_fn(src_str)
                    if has_target:
                        target_str = decode_fn(target_str)

                    if not args.quiet:
                        if src_dict is not None:
                            print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                        if has_target:
                            print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

                    # Process top predictions
                    for j, hypo in enumerate(hypos[i][:args.nbest]):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'],
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                        )
                        detok_hypo_str = decode_fn(hypo_str)
                        # if not args.quiet:
                        #     score = hypo['score'] / math.log(2)  # convert to base 2
                        #     # original hypothesis (after tokenization and BPE)
                        #     print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                        #     # detokenized hypothesis
                        #     print('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                        #     print('P-{}\t{}'.format(
                        #         sample_id,
                        #         ' '.join(map(
                        #             lambda x: '{:.4f}'.format(x),
                        #             # convert from base e to base 2
                        #             hypo['positional_scores'].div_(math.log(2)).tolist(),
                        #         ))
                        #     ), file=output_file)

                        #     if args.print_alignment:
                        #         print('A-{}\t{}'.format(
                        #             sample_id,
                        #             ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                        #         ), file=output_file)

                        #     if args.print_step:
                        #         print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                        #     if getattr(args, 'retain_iter_history', False):
                        #         for step, h in enumerate(hypo['history']):
                        #             _, h_str, _ = utils.post_process_prediction(
                        #                 hypo_tokens=h['tokens'].int().cpu(),
                        #                 src_str=src_str,
                        #                 alignment=None,
                        #                 align_dict=None,
                        #                 tgt_dict=tgt_dict,
                        #                 remove_bpe=None,
                        #             )
                        #             print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                        # Score only the top hypothesis
                        if has_target and j == 0:
                            if align_dict is not None or args.remove_bpe is not None:
                                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                                target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                                hypo_tokens = tgt_dict.encode_line(detok_hypo_str, add_if_not_exist=True)
                            if hasattr(scorer, 'add_string'):
                                scorer.add_string(target_str, detok_hypo_str)
                            else:
                                scorer.add(target_tokens, hypo_tokens)


                    if args.knn_start > -1 and knn_num_samples_proc == args.knn_proc:
                        break
                    if args.save_knn_subset and total_saved >= args.save_knn_subset_num:
                        break
                    #if i > 10:
                    #    break
                if args.knn_start > -1 and knn_num_samples_proc == args.knn_proc:
                    break
                if args.save_knn_subset and total_saved >= args.save_knn_subset_num:
                    break
                if args.knn_add_to_idx:
                    adding_to_faiss += keys.shape[0]
                    
                    index.add_with_ids(keys, addids)
                    args.all_ds_size=addids.shape
                    # for fidx in range(len(faiss_indices)):
                        
                    #     faiss_indices[fidx].add_with_ids(keys, addids)

                    #print(f"loop time {time.time()-knn_start_loop}s")

                #print(idx)
                #if idx == 0:
                #    break



                wps_meter.update(num_generated_tokens)
                progress.log({'wps': round(wps_meter.avg)})
                # num_sentences += sample["nsentences"] if "nsentences" in sample else sample['id'].numel()
        # else
            # print("NOT USING KNNMT !!")
        # print("finished creating datastore in "+str(time.time()-start_time))
        start_time = time.time()
        if args.english_centric:
            results=results+translate(src_lang='en',tgt_lang=args.final_target_lang,use_KNN=args.use_KNN,index=index)
        else:
            results=results+translate(src_lang=args.source_lang,tgt_lang=args.target_lang,use_KNN=args.use_KNN,index=index)
        # FINISHED CREATING DATASTORE !!!!!!!!
        batch_src_ds = []
        batch_tgt_ds = []
        batch_src_sent = []
        batch_tgt_sent = []
        batched = 0
        # name = (len("00000")-len(str(sent_id)))*"0"+str(sent_id)
        # test_sent_path = os.path.join(args.test_data_folder,"sent_spm_"+name+".bin")
        # test_sent_path = os.path.join(args.test_data_folder,"sent_"+name+".de")
        # args.gen_subset = "train"
        # args.knnmt=args.use_KNN
        # args.knn_add_to_idx=False
        # args.score_reference=None
        # args.no_load_keys=True
        # args.knn_add_to_idx=False
        # args.save_knn_dstore=False
        # task = tasks.setup_task(args)
        # generator = task.build_generator(models, args)
        # if args.use_KNN == False:
        #     index = None
        # # print(args.data)
        # # args.data=args.test_data
        # # task.load_dataset(args.gen_subset)
        # # itr = task.get_batch_iterator(
        # #     dataset=task.dataset(args.gen_subset),
        # #     max_tokens=args.max_tokens,
        # #     max_sentences=args.max_sentences,
        # #     max_positions=utils.resolve_max_positions(
        # #     task.max_positions(),
        # #     *[model.max_positions() for model in models]
        # #     ),
        # #     ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        # #     required_batch_size_multiple=args.required_batch_size_multiple,
        # #     num_shards=args.num_shards,
        # #     shard_id=args.shard_id,
        # #     num_workers=args.num_workers,
        # # ).next_epoch_itr(shuffle=False)
        # # progress = progress_bar.progress_bar(
        # #     itr,
        # #     log_format=args.log_format,
        # #     log_interval=args.log_interval,
        # #     default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
        # # )

        # # interactive input 

        
        
        # # test_lines = read_files(test_sent_path)
        # # import pdb;pdb.set_trace()
        # itr = make_batches(args=args,src_lines=batch_src_sent,
        #                    tgt_lines=None,models=models ,
        #                    task=task ,  encode_fn=encode_fn)
        
        batched = 0
        # progress = progress_bar.progress_bar(
        #     itr,
        #     log_format=args.log_format,
        #     log_interval=args.log_interval,
        #     default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
        # )

        
        # for idx, sample in enumerate(progress):
        #     sample = utils.move_to_cuda(sample) if use_cuda else sample
        #     if 'net_input' not in sample:
        #         continue

        #     ## For processing in parallel
        #     if args.save_knn_dstore and to_skip > 0:
        #         num_samples = sample['target'].shape[0]
        #         if to_skip - num_samples > 0:
        #             to_skip -= num_samples
        #             target_tokens = utils.strip_pad(sample['target'], tgt_dict.pad()).int().cpu()
        #             start_pos += len(target_tokens)
        #             continue

        #         for i, sample_id in enumerate(sample['id'].tolist()):
        #             if to_skip > 0:
        #                 to_skip -= 1
        #                 target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
        #                 start_pos += len(target_tokens)
        #             else:
        #                 tgt_tokens = utils.strip_pad(sample['target'][i:], tgt_dict.pad()).int().cpu()
        #                 new_sample = {
        #                         'id': sample['id'][i:],
        #                         'nsentences': len(sample['id'][i:]),
        #                         'ntokens': len(tgt_tokens),
        #                         'net_input': {
        #                             'src_tokens': sample['net_input']['src_tokens'][i:],
        #                             'src_lengths': sample['net_input']['src_lengths'][i:],
        #                             'prev_output_tokens': sample['net_input']['prev_output_tokens'][i:],
        #                         },
        #                         'target': sample['target'][i:]

        #                 }
        #                 sample = new_sample
        #                 break

        #         print('Starting the saving at location %d in the mmap' % start_pos)
        #     ## For processing in parallel

        #     prefix_tokens = None
        #     if args.prefix_size > 0:
        #         prefix_tokens = sample['target'][:, :args.prefix_size]

        #     #print('target', sample['target'].shape)
        #     # gen_timer.start()
            
        #     # if sample["ntokens"] > 15:
        #     #     import pdb
        #     #     pdb.set_trace()
        #     #     t = time.time()
        #     #     index=faiss.index_cpu_to_gpu(res, 0, index)
        #     #     print(time.time()-t)
        #     #     pdb.set_trace()
            
        #     hypos = task.inference_step(generator, models, sample, prefix_tokens,index=index)

        #     if args.save_knns_csv and args.save_knns:
        #         src_sent_decoded = sp.decode([task.tgt_dict[token] for token in sample["net_input"]["src_tokens"][0]])
        #         tgt_sent_decoded = sp.decode([task.tgt_dict[token] for token in sample["target"][0]])
        #         knns_decoded = sp.decode([task.tgt_dict[int(token)] for token in hypos[0][0]["knns"]])
        #         knn_dists = [dist for dist in hypos[0][0]["dists"]]
        #         knns_tokens = [sp.decode(task.tgt_dict[int(token)]) for token in hypos[0][0]["knns"]]
        #         pd_dic={"src_decoded":[src_sent_decoded,0],"tgt_decoded":[tgt_sent_decoded,0],"knns_decoded":[knns_decoded,0]}
        #         pd_dic.update({"token_"+str(i):[token,round(float(dist),4)] for i,(dist,token) in zip(range(len(knn_dists)),zip(knn_dists,knns_tokens))})
        #         sentence_id=str(sample["id"].item())
        #         # pd.DataFrame(pd_dic).to_csv(args.save_knns_csv+"/sentence_"+sentence_id+".csv")
        #     num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        #     #print(hypos[0][0]['tokens'])
        #     #exit(0)
        #     # gen_timer.stop(num_generated_tokens)

        #     if args.knn_add_to_idx:
        #         saving = sample['ntokens']
        #         if args.drop_lang_tok:
        #             saving = sample['ntokens'] - sample['target'].shape[0]
        #         keys = np.zeros([saving, model.decoder.embed_dim], dtype=np.float32)
        #         addids = np.zeros([saving], dtype=np.int)
        #         save_idx = 0

        #     for i, sample_id in enumerate(sample['id'].tolist()):
        #         # loop_start = time.time()
        #         has_target = sample['target'] is not None
        #         #print(sample['target'][i])

        #         # Remove padding
        #         if 'src_tokens' in sample['net_input']:
        #             src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
        #         else:
        #             src_tokens = None

        #         target_tokens = None
        #         if has_target:
        #             target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

        #         #print(len(hypos))
        #         #print(hypos[i][0]['tokens'].shape)
        #         #print(len(target_tokens))
        #         #print(hypos[i][0]['tokens'])
        #         ## knn saving code
        #         if args.save_knn_dstore:
        #             hypo = hypos[i][0]
        #             num_items = len(hypo['tokens'])
        #             #print(num_items, hypo['dstore_keys_mt'].shape)
        #             #print(hypo['tokens'])
        #             #print(hypo['dstore_keys_mt'])
        #             #exit(0)
        #             #sample_order_lens[0].append(sample_id)
        #             #sample_order_lens[1].append(num_items)
        #             #if dstore_idx + shape[0] > args.dstore_size:
        #             #    shape = [args.dstore_size - dstore_idx]
        #             #    hypo['dstore_keys_mt'] = hypo['dstore_keys_mt'][:shape[0]]
        #             if args.knn_start > -1:
        #                 if dstore_idx + num_items > dstore_keys.shape[0]:
        #                     if args.dstore_fp16:
        #                         dstore_keys = np.concatenate([dstore_keys, np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float16)], axis=0)
        #                         dstore_vals = np.concatenate([dstore_vals, np.zeros([chunk_size, 1], dtype=np.int16)], axis=0)
        #                     else:
        #                         dstore_keys = np.concatenate([dstore_keys, np.zeros([chunk_size, model.decoder.embed_dim], dtype=np.float32)], axis=0)
        #                         dstore_vals = np.concatenate([dstore_vals, np.zeros([chunk_size, 1], dtype=np.int)], axis=0)

        #             skip = 0
        #             if args.drop_lang_tok:
        #                 skip += 1

        #             if args.save_knn_subset:
        #                 if total_saved + num_items - skip > args.save_knn_subset_num:
        #                     num_items = args.save_knn_subset_num - total_saved + skip

        #             if args.knn_add_to_idx:
        #                 keys[save_idx:save_idx+num_items-skip] = hypo['dstore_keys_mt'][skip:num_items].view(
        #                         -1, model.decoder.embed_dim).cpu().numpy().astype(np.float32)
        #                 addids[save_idx:save_idx+num_items-skip] = hypo['tokens'][skip:num_items].view(
        #                         -1).cpu().numpy().astype(np.int)
        #                 save_idx += num_items - skip

        #             if not args.knn_add_to_idx:
        #                 if args.dstore_fp16:
        #                     dstore_keys[dstore_idx:num_items-skip+dstore_idx] = hypo['dstore_keys_mt'][skip:num_items].view(
        #                             -1, model.decoder.embed_dim).cpu().numpy().astype(np.float16)
        #                     dstore_vals[dstore_idx:num_items-skip+dstore_idx] = hypo['tokens'][skip:num_items].view(
        #                             -1, 1).cpu().numpy().astype(np.int16)
        #                 else:
        #                     dstore_keys[dstore_idx:num_items-skip+dstore_idx] = hypo['dstore_keys_mt'][skip:num_items].view(
        #                             -1, model.decoder.embed_dim).cpu().numpy().astype(np.float32)
        #                     dstore_vals[dstore_idx:num_items-skip+dstore_idx] = hypo['tokens'][skip:num_items].view(
        #                             -1, 1).cpu().numpy().astype(np.int)

        #             dstore_idx += num_items - skip
        #             total_saved += num_items - skip
        #             knn_num_samples_proc += 1
        #         ## knn saving code
        #         if args.score_reference:
        #             continue

        #         ## error analysis knnmt: save knns, vals and probs
        #         if args.knnmt and args.save_knns:
        #             to_save_objects.append(
        #                     {
        #                         "id": sample_id,
        #                         "src": src_tokens,
        #                         "tgt": target_tokens,
        #                         "hypo": hypos[i],
        #                     }
        #                 )
        #         ## error analysis knnmt: save knns, vals and probs

        #         # Either retrieve the original sentences or regenerate them from tokens.
        #         if align_dict is not None:
        #             src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
        #             target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
        #         else:
        #             if src_dict is not None:
        #                 src_str = src_dict.string(src_tokens, args.remove_bpe)
        #             else:
        #                 src_str = ""
        #             #print(get_symbols_to_strip_from_output(generator))
        #             if has_target:
        #                 target_str = tgt_dict.string(
        #                     target_tokens,
        #                     args.remove_bpe,
        #                     escape_unk=True,
        #                     extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
        #                 )

        #         src_str = decode_fn(src_str)
        #         if has_target:
        #             target_str = decode_fn(target_str)

        #         if not args.quiet:
        #             if src_dict is not None:
        #                 print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
        #             if has_target:
        #                 print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

        #         # Process top predictions
        #         for j, hypo in enumerate(hypos[i][:args.nbest]):
        #             hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        #                 hypo_tokens=hypo['tokens'].int().cpu(),
        #                 src_str=src_str,
        #                 alignment=hypo['alignment'],
        #                 align_dict=align_dict,
        #                 tgt_dict=tgt_dict,
        #                 remove_bpe=args.remove_bpe,
        #                 extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
        #             )
        #             sp=spm.SentencePieceProcessor(args.sentencepiece_model)
        #             detok_hypo_str = sp.decode(hypo_str.split(" "))
        #             detok_hypo_str=detok_hypo_str.replace("__en__","")
        #             results.append(detok_hypo_str)
        #             if not args.quiet:
        #             # if True:
        #                 score = hypo['score'] / math.log(2)  # convert to base 2
        #                 # original hypothesis (after tokenization and BPE)
        #                 print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
        #                 # detokenized hypothesis
        #                 print('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
        #                 print('P-{}\t{}'.format(
        #                     sample_id,
        #                     ' '.join(map(
        #                         lambda x: '{:.4f}'.format(x),
        #                         # convert from base e to base 2
        #                         hypo['positional_scores'].div_(math.log(2)).tolist(),
        #                     ))
        #                 ), file=output_file)

        #                 if args.print_alignment:
        #                     print('A-{}\t{}'.format(
        #                         sample_id,
        #                         ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
        #                     ), file=output_file)

        #                 if args.print_step:
        #                     print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

        #                 if getattr(args, 'retain_iter_history', False):
        #                     for step, h in enumerate(hypo['history']):
        #                         _, h_str, _ = utils.post_process_prediction(
        #                             hypo_tokens=h['tokens'].int().cpu(),
        #                             src_str=src_str,
        #                             alignment=None,
        #                             align_dict=None,
        #                             tgt_dict=tgt_dict,
        #                             remove_bpe=None,
        #                         )
        #                         print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)
        #             # import pdb;pdb.set_trace()
        #             # Score only the top hypothesis
        #             if has_target and j == 0:
        #                 if align_dict is not None or args.remove_bpe is not None:
        #                     # Convert back to tokens for evaluation with unk replacement and/or without BPE
        #                     target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
        #                     hypo_tokens = tgt_dict.encode_line(detok_hypo_str, add_if_not_exist=True)
        #                 if hasattr(scorer, 'add_string'):
        #                     scorer.add_string(target_str, detok_hypo_str)
        #                 else:
        #                     scorer.add(target_tokens, hypo_tokens)


        #         if args.knn_start > -1 and knn_num_samples_proc == args.knn_proc:
        #             break
        #         if args.save_knn_subset and total_saved >= args.save_knn_subset_num:
        #             break
        #         #if i > 10:
        #         #    break
        #     if args.knn_start > -1 and knn_num_samples_proc == args.knn_proc:
        #         break
        #     if args.save_knn_subset and total_saved >= args.save_knn_subset_num:
        #         break
            
        #         #print(f"loop time {time.time()-knn_start_loop}s")

        #     #print(idx)
        #     #if idx == 0:
        #     #    break
            


        #     wps_meter.update(num_generated_tokens)
        #     progress.log({'wps': round(wps_meter.avg)})
        #     num_sentences += sample["nsentences"] if "nsentences" in sample else sample['id'].numel()

        # print("finished translation in "+str(time.time()-start_time))
        # print("PROCESS TOOK "+str(time.time()-s_time_all))
        # if args.knn_q2gpu:
    # print("begining tim   e "+str(time.time()-st))
    print("missmatch number = ")
    print(args.knns_missmatch)
    print("number of correct ")
    print(args.correct_knns)
    args.results = results
    # print("time taken for ds encoding "+str(args.total_decoding_ds_time))
    return results
        #     index_ivf.quantizer = quantizer
        #     del quantizer_gpu

        # if args.save_knn_dstore:
        #     if args.knn_start > -1:
        #         dstore_keys = dstore_keys[:total_saved]
        #         dstore_vals = dstore_vals[:total_saved]
        #         np.savez(args.dstore_mmap+".keys_vals.%d.%d" % (args.knn_start, args.knn_start + knn_num_samples_proc - 1), keys=dstore_keys, vals=dstore_vals)
        #         print("Final dstore position = %d" % (start_pos + total_saved - 1))
        #         print("Number of examples processed = %d" % knn_num_samples_proc)
        #         knn_samples_savefile = args.dstore_mmap+".samples.%d.%d" % (args.knn_start, args.knn_start + knn_num_samples_proc - 1)
        #     #else:
        #     #    knn_samples_savefile = args.dstore_mmap+".samples"
        #     #np.save(knn_samples_savefile, np.array(sample_order_lens, dtype=np.int))
        #     print("dstore_idx", dstore_idx, "final number of items added", num_items - skip, "total saved", total_saved)
        #     if not args.knn_add_to_idx:
        #         print("Keys", dstore_keys.shape, dstore_keys.dtype)
        #         print("Vals", dstore_vals.shape, dstore_vals.dtype)
        #     else:
        #         for widx, write_index in enumerate(args.write_index):
        #             print("************")
        #             print(write_index)
        #             print(widx)
        #             cpu_index = faiss.index_gpu_to_cpu(faiss_indices[widx])
        #             faiss.write_index(cpu_index, write_index)
        #             # faiss.write_index(faiss_indices[widx], write_index)
        #             print("Added to faiss", adding_to_faiss)
        #             #print("Final global position %d" % global_end)

    if args.knnmt and args.save_knns:
        pickle.dump(to_save_objects, open(args.save_knns_filename, "wb"))

    # logger.info('NOTE: hypothesis and token scores are output in base 2')
    # logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
    #     num_sentences, # gen_timer.n, # gen_timer.sum, num_sentences / # gen_timer.sum, 1. / # gen_timer.avg))
    if has_target and not args.score_reference:
        if args.bpe and not args.sacrebleu:
            if args.remove_bpe:
                logger.warning("BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization")
            else:
                logger.warning("If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization")
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        if args.target_lang == 'ja':
            print("Sending sacrebleu tokenier: ja-mecab")
            print(
                'Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string(tokenize='ja-mecab')),
                file=output_file)
        elif args.target_lang == 'zh':
            print("Sending sacrebleu tokenier: zh")
            print(
                'Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string(tokenize='zh')),
                file=output_file)
        else:
            print(
                'Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()),
                file=output_file)

    return scorer


def cli_main(args=None,models=None):
    if args and models:
        main(args,models)

    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    args.sentencepiece_model="/mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/vocab/deen_spm/sentencepiece.bpe.model"
    args.target_lang="en"
    args.source_lang="de"
    args.model_overrides= "{'knn_keytype':'last_ffn_input'}"
    args.save_knns_csv="/home/ababouelenin/CustomTranslation/KNN-MT/save_knns_detailed/1m_datastore_k_1_beam_1"
    main(args)


if __name__ == '__main__':
    cli_main()
