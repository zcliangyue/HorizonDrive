import os
import torch
import torch.distributed as dist

from .fsdp import shard_model

try:
    try:
        import pai_fuser
        from pai_fuser.core.distributed import (
            get_sequence_parallel_rank, get_sequence_parallel_world_size,
            get_sp_group, get_world_group, init_distributed_environment,
            initialize_model_parallel)
        from pai_fuser.core.long_ctx_attention import \
            xFuserLongContextAttention
        print("Enable PAI DiT Turbo")
    except Exception as ex:
        import xfuser
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group, get_world_group,
                                             init_distributed_environment,
                                             initialize_model_parallel)
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except Exception as ex:
    xFuserLongContextAttention = None

    class _TorchSequenceParallelGroup:
        def all_gather(self, tensor, dim=0):
            tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensors, tensor.contiguous())
            return torch.cat(tensors, dim=dim)

    class _TorchWorldGroup:
        @property
        def rank(self):
            return dist.get_rank() if dist.is_initialized() else 0

        @property
        def local_rank(self):
            return int(os.environ.get("LOCAL_RANK", self.rank))

    _torch_sp_group = _TorchSequenceParallelGroup()

    def get_sequence_parallel_world_size():
        return dist.get_world_size() if dist.is_initialized() else 1

    def get_sequence_parallel_rank():
        return dist.get_rank() if dist.is_initialized() else 0

    def get_sp_group():
        return _torch_sp_group

    def get_world_group():
        return _TorchWorldGroup()

    def init_distributed_environment(rank, world_size):
        return None

    def initialize_model_parallel(sequence_parallel_degree, ring_degree, ulysses_degree):
        return None

try: 
    from pai_fuser.core import parallel_magvit_vae
    print("Enable PAI VAE Turbo")
except:
    def parallel_magvit_vae(multi_gpus_overlap_scale, spatial_compression_ratio):
        def decorator(func):
            def wrapper(self, z, *args, **kwargs):
                decoded = func(self, z, *args, **kwargs)
                return decoded
            return wrapper
        return decorator

def set_multi_gpus_devices(ulysses_degree, ring_degree):
    if ulysses_degree > 1 or ring_degree > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        print('parallel inference enabled: ulysses_degree=%d ring_degree=%d rank=%d world_size=%d' % (
            ulysses_degree, ring_degree, dist.get_rank(),
            dist.get_world_size()))
        assert dist.get_world_size() == ring_degree * ulysses_degree, \
                    "number of GPUs(%d) should be equal to ring_degree * ulysses_degree." % dist.get_world_size()
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree)
        # device = torch.device("cuda:%d" % dist.get_rank())
        device = torch.device(f"cuda:{get_world_group().local_rank}")
        print('rank=%d device=%s' % (get_world_group().rank, str(device)))
    else:
        device = "cuda"
    return device
