# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
import numpy as np
import time

from fairseq import utils
from fairseq.data import Dictionary
import sentencepiece as spm
import torch
# from transformers import NllbTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(
        self, tgt_dict, softmax_batch=None, compute_alignment=False, eos=None,
        symbols_to_strip_from_output=None, args=None,spm=None,
    ):

        self.pad = tgt_dict.pad()
        self.args=args
        self.use_pretrained = args.use_pretrained
        # import sentencepiece as spm
        self.eos = tgt_dict.eos() if eos is None else eos
        self.tgt_dict=tgt_dict
        self.sp = args.sp
        self.softmax_batch = softmax_batch or sys.maxsize
        # self.pretained_tokenizer  = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        if args.use_pretrained :
            self.pretrained_model = args.pretrained_model
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.args = args
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None else {self.eos})

    
    @torch.no_grad()
    def mean_pooling(self,token_embeddings, mask,tgt=False):
        
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        if tgt:
            return token_embeddings
        else:
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    @torch.no_grad()
    def generate_from_pretrained(self,sample):
        src_sentences = []
        tgt_sentences = []
        for i in range(len(sample["net_input"]["src_tokens"])):
            src_sent_decoded = self.sp.decode([self.tgt_dict[token] for token in sample["net_input"]["src_tokens"][i]]).replace("<pad>","")
            # tgt_sent_decoded =  self.sp.decode([self.tgt_dict[token] for token in sample["target"][i]]).replace("<pad>","")
            src_sentences.append(src_sent_decoded)
            # tgt_sentences.append(tgt_sent_decoded)
        
        src_embeddings = self.pretrained_model.encode(src_sentences,show_progress_bar=False,batch_size=4)
        # tgt_embeddings=self.pretrained_model.encode(tgt_sentences,batch_size=len(tgt_sentences))
        return src_embeddings , None , src_sentences , None
        return src_embeddings , tgt_embeddings , src_sentences , tgt_sentences
        # self.pretained_tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
        # src_tokenized = self.pretained_tokenizer(src_sentences, padding=True, truncation=True, return_tensors='pt')
        # src_embeddings = self.pretrained_model(**src_tokenized)
        # src_sentence_embeddings = self.mean_pooling(src_embeddings['last_hidden_state'], src_tokenized['attention_mask'])

        # tgt_tokenized = self.pretained_tokenizer(tgt_sentences, padding=True, truncation=True, return_tensors='pt')
        # tgt_embeddings = self.pretrained_model(**tgt_tokenized)
        # # tgt_sentence_embeddings = self.mean_pooling(tgt_embeddings['last_hidden_state'], tgt_tokenized['attention_mask'])
        # final_embeddings = (src_sentence_embeddings.unsqueeze(dim=1)+tgt_embeddings[0])/2

        # return src_sentence_embeddings,final_embeddings
        

        
    

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']
        pretrained_src_sent_embeddings=None
        if self.use_pretrained:
            pretrained_src_sent_embeddings,pretrained_tgt_embeddings \
            , src_sentences , tgt_sentences =self.generate_from_pretrained(sample)

        

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
            combine_probs = torch.stack([vocab_p, knn_p], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - coeff)
            coeffs[1] = np.log(coeff)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

            return curr_prob

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            # tgt_token_embeddings= model.encoder.embed_tokens(sample['target'])
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for i, (bd, tgt, is_single) in enumerate(batched):
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data

                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)

            if 'knn_dstore' in kwargs:
                dstore = kwargs['knn_dstore']
                # TxBxC
                queries = bd[1][self.args.knn_keytype]
                if len(models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                yhat_knn_prob = dstore.get_knn_log_prob(
                        queries,
                        orig_target.permute(1, 0),
                        pad_idx=self.pad)
                yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2).squeeze(-1)
                if self.args.fp16:
                    yhat_knn_prob = yhat_knn_prob.half()
                    probs = probs.half()

                probs = combine_knn_and_vocab_probs(
                            yhat_knn_prob, probs, self.args.lmbda)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            # src_sent_decoded = sp.decode([self.tgt_dict[token] for token in sample["net_input"]["src_tokens"][i]]).replace("<pad>","")
            # tgt_sent_decoded =  sp.decode([self.tgt_dict[token] for token in sample["target"][0]]).replace("<pad>","")
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            if self.args.use_pretrained == False:
                final_embeddings=decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:]
                # final_sents=[]
                # tgt_sentences=[]
                # src_sentences=[]
            else:
                final_sents,tokens,pre_tgt_embeddings = self.decode_sents(ref)
                broadcasted_shape=[decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:].shape[0],pre_tgt_embeddings.shape[1]]
                # import pdb;pdb.set_trace()
                pre_tgt_broadcasted=torch.zeros(broadcasted_shape)+ pre_tgt_embeddings[-1,:] 
                pre_tgt_broadcasted[:pre_tgt_embeddings.shape[0],:]=torch.tensor(pre_tgt_embeddings)
                pre_embed_boradcasted=torch.zeros(broadcasted_shape)+pretrained_src_sent_embeddings[i]
                pre_embeddings=torch.cat((pre_embed_boradcasted,pre_tgt_broadcasted),dim=1)
                # pre_embeddings=(pre_tgt_broadcasted+pre_embed_boradcasted)/2
                # pre_embeddings=torch.zeros_like(pre_embeddings)
                # pre_embeddings=torch.rand_like(pre_embeddings)
                final_embeddings=pre_embeddings
                # final_embeddings=torch.cat((decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:].detach().cpu(),pre_embeddings),dim=1)
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,   
                'dstore_keys': decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:] if hasattr(self.args, 'save_knnlm_dstore') and self.args.save_knnlm_dstore else None,
                'dstore_keys_mt': final_embeddings if hasattr(self.args, 'save_knn_dstore') and self.args.save_knn_dstore else None,
                # 'src_sentence':src_sentences[i],
                # "tokens_sents":final_sents,
                # "tokens_sorted":tokens
            }])
        return hypos
    
    def decode_sents(self,ref):
        list_of_tokens=[self.tgt_dict[token] for token in ref]
        final_sents = []
        tokens_consumed=[]
        for token in list_of_tokens:
            decoded_string=self.args.sp.Decode(tokens_consumed)
            final_sents.append((decoded_string).replace('T',' ').strip(self.args.special_char))
            tokens_consumed.append(token)
        final_embeddings = self.args.pretrained_model.encode(final_sents,show_progress_bar=False,batch_size=4)

        return final_sents ,[self.tgt_dict[token] for token in ref], final_embeddings

        

