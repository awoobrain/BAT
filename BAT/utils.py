"""
copyright:
We follow the code from https://github.com/wzhouad/NLL-IE to build training data and revise sentence level to document level.
The reason is good reproducibility, easy to maintain, and easy to demo.
"""


from transformers import PreTrainedTokenizerFast
from typing import List, Optional
import re
import truecase


class CONLLDataProcessor:
    """
    Transfer words into token ids, and build targets.
    """

    def __init__(self, data_files: list, tokenizer: PreTrainedTokenizerFast, max_seq_length: Optional[int]=None):
        """
        Args:
            data_files: list
            tokenizer: PreTrainedTokenizerFast
            max_seq_length: Optional[int]
                must over 300.
                defualt: 512
        """

        self.data_files       = data_files
        self.tokenizer        = tokenizer
        self.max_seq_length   = max_seq_length or 512
        self._label_to_target = {
            "None"  :-1, 
            "O"     : 0, 
            "B-MISC": 1, 
            "I-MISC": 2, 
            "B-PER" : 3,
            "I-PER" : 4, 
            "B-ORG" : 5, 
            "I-ORG" : 6, 
            "B-LOC" : 7, 
            "I-LOC" : 8
        }

    @property
    def label_to_target(self):
        return self._label_to_target

    def generate_token_ids(self, words: list, targets: list, tokenizer: PreTrainedTokenizerFast) -> dict:
        """
        Enter words and generate token ids

        Args:
            words: list
            targets: list
            tokenizer: PreTrainedTokenizerFast
        
        Returns:
            output: dict
        """
        
        all_tokens, all_targets = [], []
        for word, target in zip(words, targets):
            tokens  = tokenizer.tokenize(word)
            targets = [self.label_to_target[target]] + [-1] * (len(tokens) - 1)
            all_tokens  += tokens
            all_targets += targets
        
        assert len(all_tokens)     == len(all_targets)
        assert len(all_tokens) + 2 <= self.max_seq_length
        all_token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        all_token_ids = tokenizer.build_inputs_with_special_tokens(all_token_ids)
        all_targets   = [-1] + all_targets + [-1]
        
        assert len(all_token_ids) == len(all_targets)
        output = {
            "token_ids"    : all_token_ids,
            "token_targets": all_targets
        }  
        return output

    def correct_sentence_case(self, words: list) -> list:
        """
        Correct sentence case. 
        
        Args:
            words: list 
        
        Returns:
            words: list
        """

        english_words = [(index, word) for index, word in enumerate(words) if word.isalpha()]
        uppercase     = [word for _, word in english_words if re.match(r"\b[A-Z\.\-]+\b",word)]

        if len(uppercase) == len(english_words) and len(english_words) > 0:
            truecase_words = truecase.get_true_case(" ".join(uppercase)).split()

            assert len(truecase_words) == len(uppercase)
            if len(truecase_words) == len(english_words):
                for (index, word), truecase_word in zip(english_words, truecase_words):
                    words[index] = truecase_word
                return words
        return words


    def read_conll_with_document_level(self, file_path: str, tokenizer: PreTrainedTokenizerFast) -> list:
        """
        Read and split document, and add [SEP] between sentences.

        Args:
            file_path: str 
            tokenizer: PreTrainedTokenizerFast

        Returns:
            samples: list
        """

        is_first_sentence = False
        samples           = []
        with open(file_path, "r") as fp:
            lines          = fp.readlines()
            words, targets = [], []
            
            buffer_words, buffer_targets, buffer_count = [], [], 0
            
            words_count = 0
            sep_token   = tokenizer.convert_ids_to_tokens(tokenizer.sep_token_id)
            for line in lines:
                line = line.strip()

                if buffer_count >= self.max_seq_length:
                    raise Exception(f"buffer is {buffer_count}, over {self.max_seq_length}")

                # enter a new document.
                if "-DOCSTART-" in line:
                    is_first_sentence = True
                    if len(words) > 0:
                        samples.append(self.generate_token_ids(words=words, targets=targets, tokenizer=tokenizer))
                        if buffer_words[0] == sep_token:
                            buffer_words.pop(0)
                            buffer_targets.pop(0)
                            buffer_count = buffer_count-1
                        
                    words, targets = [], []
                    words_count = 0
                    continue
                
                # enter to next sentence.
                if len(line) == 0:
                    if len(buffer_words) > 0 and is_first_sentence:
                        
                        # correct all capital words in the sentence.(use truecase)
                        buffer_words      = self.correct_sentence_case(buffer_words)
                        is_first_sentence = False

                    # check the words count smaller than (self.max_seq_length-2)
                    if words_count <= (self.max_seq_length - 2):

                        # if the words count + buffer count <= (self.max_seq_length-2), add buffer.
                        if words_count + buffer_count <= (self.max_seq_length - 2):
                            words_count += buffer_count
                            words       += buffer_words
                            targets     += buffer_targets
                            
                            buffer_words, buffer_targets, buffer_count = [], [], 0
                        
                        # process the words
                        else:
                            samples.append(self.generate_token_ids(words=words, targets=targets, tokenizer=tokenizer))
                            if buffer_words[0] == sep_token:
                                buffer_words.pop(0)
                                buffer_targets.pop(0)
                                buffer_count = buffer_count - 1
                            words, targets = [], []
                            words_count    = 0
                    
                    if not is_first_sentence:
                        buffer_words   += [sep_token]
                        buffer_targets += ["None"]
                        buffer_count   += 1

                elif len(line) > 0:
                    data = line.split()
                    buffer_words.append(data[0])
                    buffer_targets.append(data[-1])
                    buffer_count += len(tokenizer.tokenize(data[0]))
            words   += buffer_words
            targets += buffer_targets

            # process the rest of the words
            if len(words) > 0:
                samples.append(self.generate_token_ids(words=words, targets=targets, tokenizer=tokenizer))
        return samples

    def start(self) -> List[dict]:
        """
        Run data transformation.

        Returns:
            results: List[dict]
        """
        
        results = []
        for data_file in self.data_files:
            results += self.read_conll_with_document_level(file_path=data_file, tokenizer=self.tokenizer)
        return results