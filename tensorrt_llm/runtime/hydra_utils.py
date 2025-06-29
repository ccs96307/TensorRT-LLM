import copy
from argparse import Namespace
from functools import cmp_to_key
from typing import List

import numpy as np
import torch

from tensorrt_llm.logger import logger


def path_sorter(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return -1 if a[i] < b[i] else 1
    return 0  # shouldn't reach


path_sorting_key = cmp_to_key(path_sorter)


def expand_choices_if_needed(hydra_choices: List[List[int]]):
    """
    Do a simple check to see if the given choices are path-only or vanilla.
    """
    assert len(hydra_choices) > 0
    for c in hydra_choices:
        if len(c) > 1:
            try:
                _ = hydra_choices.index(
                    [c[0]])  # find the first parent of current path
                logger.debug(
                    "Detected vanilla-style of Hydra choices. No need to expand."
                )
                return hydra_choices  # if found, just return assuming it is already expanded
            except ValueError:
                logger.debug(
                    "Detected path-only style of Hydra choices. Expanding ...")
                break
    expanded_choices = set()
    for c in hydra_choices:
        cur = ()
        for n in c:
            cur = (*cur, n)
            expanded_choices.add(cur)
    expanded_choices = [list(c) for c in expanded_choices]
    return expanded_choices


def get_packed_mask(num_hydra_tokens, hydra_mask, max_hydra_tokens=None):
    max_hydra_tokens = num_hydra_tokens if max_hydra_tokens is None else max_hydra_tokens
    num_packed_masks = (max_hydra_tokens + 1 + 32 - 1) // 32
    hydra_packed_mask = torch.zeros((num_hydra_tokens + 1, num_packed_masks),
                                    dtype=torch.int32)
    for token_idx in range(num_hydra_tokens + 1):
        if token_idx == 0:
            hydra_packed_mask[0, 0] = 1
        else:
            mask_list = hydra_mask[token_idx - 1, :].tolist()
            # insert 1 as there is one extra new token from the original lm head.
            mask_list.insert(0, True)
            # convert binary bits into 4 int32_t
            mask_str_list = [str(int(val)) for val in mask_list]
            mask_str_list.reverse()

            for mask_idx in range(num_packed_masks):
                if mask_idx * 32 >= len(mask_str_list):
                    break
                mask_32bits_str = ''.join(mask_str_list[-(mask_idx + 1) * 32:
                                                        (-mask_idx * 32 - 1)] +
                                          [mask_str_list[(-mask_idx * 32 - 1)]])
                valid_num_bits = len(mask_32bits_str)
                first_bit1 = mask_32bits_str[0] == '1'
                mask_31bits_str = mask_32bits_str[1:]
                mask_31bits = 0 if mask_31bits_str == "" else int(
                    mask_31bits_str, 2)
                if valid_num_bits == 32:
                    mask_32bits = mask_31bits - first_bit1 * (2**(
                        valid_num_bits - 1))
                else:
                    mask_32bits = mask_31bits + first_bit1 * (2**(
                        valid_num_bits - 1))
                hydra_packed_mask[token_idx, mask_idx] = mask_32bits
    return hydra_packed_mask


def choices_2_paths(num_hydra_heads, choices):
    paths = {}
    all_paths = {}
    level_counts = [0] * num_hydra_heads
    choices.sort(key=len, reverse=True)
    for c in choices:
        k = ":".join([str(ci) for ci in c])
        if k not in all_paths:
            paths[k] = c
        for i in range(len(c)):
            k = ":".join([str(ci) for ci in c[:i + 1]])
            if k not in all_paths:
                all_paths[k] = c[:i + 1]
                level_counts[i] += 1
    return list(paths.values()), level_counts, paths, all_paths


def get_hydra_topks(num_hydra_heads, paths):
    hydra_topks = [0] * num_hydra_heads
    for p in paths:
        for i, k in enumerate(p):
            hydra_topks[i] = max(hydra_topks[i], k + 1)
    return hydra_topks


def get_hydra_tree(num_hydra_heads, hydra_topks, level_counts, paths):
    cum_topks = np.cumsum([0] + hydra_topks)
    cum_level_counts = np.cumsum([0] + level_counts)
    tree_paths = copy.deepcopy(paths)
    hydra_tree_ids = list(np.arange(hydra_topks[0]))
    hydra_position_offsets = [0] * hydra_topks[0]
    for i in range(1, num_hydra_heads):
        last_prefix = "-1"
        last = -1
        c = -1
        for pi, p in enumerate(paths):
            if i < len(p):
                prefix_str = ":".join([str(k) for k in p[:i]])
                if last_prefix != prefix_str or last != p[i]:
                    # new path
                    hydra_position_offsets.append(i)
                    hydra_tree_ids.append(p[i] + cum_topks[i])
                    last_prefix = prefix_str
                    last = p[i]
                    c += 1
                tree_paths[pi][i] = cum_level_counts[i] + c
    return hydra_tree_ids, hydra_position_offsets, tree_paths


def get_hydra_mask(hydra_tree_ids, hydra_paths):
    hydra_mask = torch.zeros((len(hydra_tree_ids), len(hydra_tree_ids)))
    hydra_mask[:, 0] = 1
    for p in hydra_paths:
        for i, idx in enumerate(p):
            if idx < 0:
                continue
            for j in range(i + 1):
                hydra_mask[idx, p[j]] = 1
    return hydra_mask


def _hydra_setup(choices_or_paths, num_hydra_heads=None):
    choices = copy.deepcopy(choices_or_paths)
    sorted_choices = sorted(choices, key=path_sorting_key)
    if num_hydra_heads is None:
        num_hydra_heads = max([len(c) for c in sorted_choices])
    paths, level_counts, _, _ = choices_2_paths(num_hydra_heads, sorted_choices)
    paths = sorted(paths, key=path_sorting_key)
    hydra_topks = get_hydra_topks(num_hydra_heads, paths)
    hydra_tree_ids, hydra_position_offsets, tree_paths = get_hydra_tree(
        num_hydra_heads, hydra_topks, level_counts, paths)

    num_hydra_tokens = len(hydra_tree_ids)
    # now do the padding before converting to torch.Tensor
    hydra_paths = []
    for p in tree_paths:
        hydra_paths.append(
            torch.tensor([-1] + p + ([-2] * (num_hydra_heads - len(p)))))
    hydra_topks = torch.tensor(hydra_topks)
    hydra_paths = torch.stack(hydra_paths) + 1
    hydra_tree_ids = torch.tensor([-1] + hydra_tree_ids) + 1
    hydra_position_offsets = torch.tensor([-1] + hydra_position_offsets) + 1
    hydra_mask = get_hydra_mask(hydra_tree_ids, hydra_paths)
    hydra_packed_mask = get_packed_mask(num_hydra_tokens, hydra_mask[1:, 1:])
    hydra_spec_decoding_use = torch.tensor([1], device="cpu")

    return Namespace(
        hydra_mask=hydra_mask.cuda(),
        hydra_packed_mask=hydra_packed_mask.cuda(),
        hydra_topks=hydra_topks.cuda(),
        hydra_paths=hydra_paths.cuda(),
        hydra_tree_ids=hydra_tree_ids.cuda(),
        hydra_position_offsets=hydra_position_offsets.cuda(),
        hydra_spec_decoding_use=hydra_spec_decoding_use,
    )
