import re
import os, shutil
import json
from tokenizers import ByteLevelBPETokenizer
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, vocab_size=None, force=True, tokenizer_type='character_wise', **kwargs):
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
            self.ind2token = dict(enumerate(self.vocab))
            self.token2ind = {v: k for k, v in self.ind2char.items()}

        # REDO:
        elif tokenizer_type == 'bpe':
            PATH = './data/datasets/librispeech/train-clean-100_index.json'
            TEMP_CORPUS_PATH = './src/text_encoder/bpe/temp_corpus.txt'
            MODEL_PATH = './src/text_encoder/bpe/'
            special_tokens = ["<unk>", "EMPTY_TOKEN"]

            if force:
                shutil.rmtree(MODEL_PATH)
                os.mkdir(MODEL_PATH)

            with open(PATH) as f:
                data = json.load(f)
                corpus = '\n'.join([el['text'] for el in data])
                
            with open(TEMP_CORPUS_PATH, 'w+') as f:
                f.write(corpus)

            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(files=TEMP_CORPUS_PATH, vocab_size=vocab_size, show_progress=True, special_tokens=special_tokens)
            tokenizer.save_model(MODEL_PATH, 'bpe_tokenizer_new')

            os.remove(TEMP_CORPUS_PATH)

            self.tokenizer = ByteLevelBPETokenizer(
                f"{MODEL_PATH}bpe_tokenizer_new-vocab.json",
                f"{MODEL_PATH}bpe_tokenizer_new-merges.txt",
            ) 

            self.vocab = self.tokenizer.get_vocab()
            
    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        if self.tokenizer_type == 'character_wise':
            return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            if self.tokenizer_type == 'character_wise':
                return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
            else:
                # print(text)
                tokenized = self.tokenizer.encode(text).ids
                # print(torch.tensor(tokenized))
                return torch.tensor(tokenized)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
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
        if self.tokenizer_type == 'character_wise':
            return "".join([self.ind2char[int(ind)] for ind in inds]).strip()
        else:
            return self.tokenizer.decode(inds)

    def ctc_decode(self, inds) -> str:
        # ind -> chars
        res_list = []

        last_ind = None
        for i in range(len(inds)):
            if  i == 0 or inds[i] != inds[i - 1] and \
                (self.tokenizer_type == 'character_wise' and inds[i] != self.char2ind[self.EMPTY_TOK] or \
                self.tokenizer_type == 'bpe' and inds[i] != self.tokenizer.token_to_id(self.EMPTY_TOK)):
                res_list.append(inds[i])

        return self.decode(res_list)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
