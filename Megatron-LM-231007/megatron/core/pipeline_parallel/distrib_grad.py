# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import mpu
from megatron.core.utils import get_attr_wrapped_model, get_model_config


def _allreduce_word_embedding_grads(model, config):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings
    parameters stay in sync. This should only run for models that support
    pipelined model parallelism (BERT and GPT-2).
    """

    if (
        mpu.is_rank_in_embedding_group(ignore_virtual=True)
        and mpu.get_pipeline_model_parallel_world_size() > 1
    ):
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif mpu.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad = weight.main_grad
            torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())


def _allreduce_position_embedding_grads(model, config):
    """
    All-reduce position_embeddings grad across first (encoder) and
    split (decoder) stages to ensure that position embeddings parameters
    stay in sync. This should only run for T5 models with pipeline
    parallelism.
    """
    if (
        mpu.is_rank_in_position_embedding_group()
        and mpu.get_pipeline_model_parallel_world_size() > 1
        and config.pipeline_model_parallel_split_rank is not None
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )
        torch.distributed.all_reduce(grad, group=mpu.get_position_embedding_group())


def _allreduce_embedding_grads(model, config):
    """All-reduce both word and position embeddings."""
    _allreduce_word_embedding_grads(model, config)
    _allreduce_position_embedding_grads(model, config)


def _allreduce_layernorm_grads(model, config):
    """All-reduce layernorm grads (for sequence parallelism)."""

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if mpu.get_tensor_model_parallel_world_size() > 1 and config.sequence_parallel:
        grads = []
        for model_chunk in model:
            for param in get_attr_wrapped_model(model_chunk, 'parameters')():
                if getattr(param, 'sequence_parallel', False):
                    grad = param.main_grad
                    grads.append(grad.data)
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=mpu.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)


def finalize_model_grads(model):
    """All-reduce all grads across DP replicas, layernorm grads
    for sequence parallelism, and embedding grads across first and
    last pipeline stages (if not tied)."""

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.sync_gradients()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads.
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()
