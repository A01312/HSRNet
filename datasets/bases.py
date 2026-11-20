from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer, basic_clean, whitespace_clean
from prettytable import PrettyTable
import random
import numpy as np
import os




class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("HSRNet.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result
 

def dependency_inputs(captions, tokenizer, model, text_length, truncate):
    if model is None:
        return None
    texts = list(dict.fromkeys(captions))
    normalized = (whitespace_clean(basic_clean(text)).lower() for text in texts)
    cached = {}
    for text, doc in zip(texts, model.pipe(normalized, batch_size=128)):
        words = torch.full((text_length,), -1, dtype=torch.long)
        parents = torch.full_like(words, -1)
        matched_words, cursor = [], 0
        for match in tokenizer.pat.finditer(doc.text):
            while cursor < len(doc) and doc[cursor].idx + len(doc[cursor]) <= match.start():
                cursor += 1
            candidates = []
            for idx in range(cursor, len(doc)):
                if doc[idx].idx >= match.end():
                    break
                if not doc[idx].is_space:
                    candidates.append(idx)
            if not candidates:
                raise ValueError(f'Unaligned caption token: {match.group()}')
            word = max(candidates, key=lambda idx: min(doc[idx].idx + len(doc[idx]), match.end())
                                                  - max(doc[idx].idx, match.start()))
            token = match.group()
            token = ''.join(tokenizer.byte_encoder[byte] for byte in token.encode('utf-8'))
            matched_words.extend([word] * len(tokenizer.bpe(token).split(' ')))
            if len(matched_words) >= text_length - 2:
                break
        matched_words = matched_words[:text_length - 2]
        nodes = {idx: pos for pos, idx in enumerate(dict.fromkeys(matched_words))}
        for pos, idx in enumerate(matched_words, start=1):
            words[pos] = nodes[idx]
        for idx, node in nodes.items():
            parents[node] = nodes.get(doc[idx].head.i, -1)
        tokens = tokenize(text, tokenizer, text_length, truncate)
        if int(tokens.argmax()) != len(matched_words) + 1:
            raise ValueError('Dependency positions and CLIP tokens do not match')
        cached[text] = tokens, words, parents
    return cached


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None, allow_fallback=True):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform
        self.allow_fallback = allow_fallback
        
    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path, allow_fallback=self.allow_fallback)
        
 
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True,
                 dependency_model=None):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.dependencies = dependency_inputs(captions, self.tokenizer, dependency_model, text_length, truncate)
  
    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        if self.dependencies is not None:
            return pid, self.dependencies[caption]
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset, args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 dependency_model=None):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.img_aug = args.img_aug
        self.real_correspondences = np.ones(len(dataset))
        self.tokenizer = SimpleTokenizer()
        self.allow_fallback = not args.use_occlusion
        if self.txt_aug and dependency_model is not None:
            raise ValueError('Text augmentation changes the dependency alignment')
        self.dependencies = dependency_inputs((row[3] for row in dataset), self.tokenizer,
                                              dependency_model, text_length, truncate)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path, allow_fallback=self.allow_fallback)

        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = (self.dependencies[caption][0] if self.dependencies is not None
                          else tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate))
        if self.txt_aug:
            caption_tokens = self.txt_data_aug(caption_tokens.cpu().numpy())
        
        ret = {
        'pids': pid,
        'image_ids': image_id,
        'images': img,
        'caption_ids': caption_tokens,
        'index':index,
        'img_paths': img_path,
        }
        if self.dependencies is not None:
            ret['words'], ret['parents'] = self.dependencies[caption][1:]

        return ret

    def txt_data_aug(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.20:
                    prob /= 0.20
                    # 50% randomly change token to mask token
                    if prob < 0.6:
                        aug_tokens.append(mask) 
                    # 20% randomly change token to random token
                    elif prob < 0.8:
                        aug_tokens.append(random.choice(token_range)) # -> rest 10% randomly keep current token
                    else:
                        None # # 30% randomly remove
                else:
                    # no masking token (will be ignored by loss function later)
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens)
