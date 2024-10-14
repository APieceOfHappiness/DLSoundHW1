import re
import os, shutil
import json
from string import ascii_lowercase
from collections import defaultdict
import sentencepiece as spm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import ByteLevelBPETokenizer

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, vocab_size=None, force=True, tokenizer_type='character_wise', model_path=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.tokenizer_type = tokenizer_type

        if tokenizer_type == 'character_wise':
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
            self._ind2char = dict(enumerate(self.vocab))
            self._char2ind = {v: k for k, v in self._ind2char.items()}

        elif tokenizer_type == 'bpe_char':
            # print('YEEEEEE' * 1000)
            PATH = './data/datasets/librispeech/train-clean-100_index.json'
            TEMP_CORPUS_PATH = './src/text_encoder/bpe_char/temp_corpus.txt'
            MODEL_PATH = './src/text_encoder/bpe_char/'
            special_tokens = [""]

            # if force:
            #     shutil.rmtree(MODEL_PATH)
            #     os.mkdir(MODEL_PATH)

            # print('first')
            # with open(PATH) as f:
            #     data = json.load(f)
            #     corpus = '\n'.join([el['text'] for el in data])

            # print('second')
            # with open(TEMP_CORPUS_PATH, 'w+') as f:
            #     f.write(corpus)

            # self.tokenizer = Tokenizer(BPE(unk_token="[UNK]", dropout=0.1))
            # bpe_trainer = BpeTrainer(vocab_size=1000)
            # self.tokenizer.pre_tokenizer = Whitespace()
            # self.tokenizer.train(files=[TEMP_CORPUS_PATH], trainer=bpe_trainer)
            # self.tokenizer.add_special_tokens(special_tokens)
            # self.tokenizer.save(f'{MODEL_PATH}/bpe_char_tokenizer.json')
            # # self.tokenizer = Tokenizer.from_file(f'{MODEL_PATH}/bpe_char_tokenizer.json')
            # self.vocab = self.tokenizer.get_vocab()

        elif tokenizer_type == 'bpe_byte':
            # PATH = './data/datasets/librispeech/train-clean-100_index.json'
            TEMP_CORPUS_PATH = './src/text_encoder/bpe_char/temp_corpus.txt'
            MODEL_PATH = './src/text_encoder/bpe_byte/'
            special_tokens = [""]

            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(files=TEMP_CORPUS_PATH, vocab_size=vocab_size, show_progress=True, special_tokens=special_tokens)
            tokenizer.save_model(MODEL_PATH, 'bpe_tokenizer_new')

            if model_path is None:
                model_path = './src/text_encoder/bpe_byte/'

            self.tokenizer = ByteLevelBPETokenizer(
                f"{model_path}bpe_tokenizer_new-vocab.json",
                f"{model_path}bpe_tokenizer_new-merges.txt",
            ) 

            self.vocab = self.tokenizer.get_vocab()
        
    def ind2token(self, ind):
        if self.tokenizer_type == 'character_wise':
            return self._ind2char[ind]
        else:
            return self.tokenizer.id_to_token(ind) 
            # print(ind)
            # return out if out != '<EMPTY>' else self.EMPTY_TOK
        
    def token2ind(self, token):
        if self.tokenizer_type == 'character_wise':
            return self._char2ind[token]
        else:
            return self.tokenizer.token_to_id(token)

    def text2ind(self, text):
        if self.tokenizer_type == 'character_wise':
            return torch.Tensor([self.token2ind(token) for token in text]).unsqueeze(0)
        else:
            # print(text)
            return torch.Tensor(self.tokenizer.encode(text).ids)

    def ind2text(self, inds):
        if self.tokenizer_type == 'character_wise':
            return "".join([self.ind2token(int(ind)) for ind in inds]).strip()
        else:
            # print(inds)
            return self.tokenizer.decode(inds)
            
    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2token(item)

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return self.text2ind(text)
        except KeyError:
            raise Exception(
                f"Can't encode text '{text}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return self.ind2text(inds)
    
    def ctc_decode(self, inds) -> str:
        # ind -> chars
        res_list = []

        last_ind = None
        for i in range(len(inds)):
            if i == 0 or inds[i] != inds[i - 1] and inds[i] != self.token2ind(self.EMPTY_TOK):
                res_list.append(inds[i])

        return self.decode(res_list)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def expand_and_merge_path(self, dp, next_token_probs, ind2token):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_ind = ind
            for (prefix, last_ind), v in dp.items():
                if last_ind == cur_ind:
                    new_prefix = prefix
                else:
                    if cur_ind != self.token2ind(self.EMPTY_TOK):
                        new_prefix = prefix + (cur_ind,)
                    else:
                        new_prefix = prefix
                new_dp[new_prefix, cur_ind] += v * next_token_prob
        return new_dp

    def truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: x[1])[-beam_size:])

    def ctc_beam_search_ind(self, probs, beam_size):
        dp = {
            ((self.token2ind(''),), (self.token2ind(self.EMPTY_TOK),)): 1.0,
        }
        for prob in probs:
            dp = self.expand_and_merge_path(dp, prob, self.ind2token)
            dp = self.truncate_paths(dp, beam_size)
        dp = [(prefix, proba) for (prefix, _), proba in sorted(dp.items(), key=lambda x: x[1])][-beam_size:]
        return dp
