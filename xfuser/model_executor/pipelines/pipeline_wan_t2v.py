from typing import Callable, Optional

from diffusers import WanPipeline

from xfuser.core.distributed import get_runtime_state


class xFuserWanT2VPipeline(WanPipeline):
    """Thin xDiT wrapper around WanPipeline that hooks fp8_comms sync into the denoising loop."""

    def __call__(self, *args, callback_on_step_end: Optional[Callable] = None, **kwargs):
        def _sync_and_forward(pipeline, i, t, callback_kwargs):
            get_runtime_state().sync_fp8_comms(pipeline.transformer)
            if callback_on_step_end is not None:
                return callback_on_step_end(pipeline, i, t, callback_kwargs)
            return callback_kwargs

        return super().__call__(*args, callback_on_step_end=_sync_and_forward, **kwargs)
