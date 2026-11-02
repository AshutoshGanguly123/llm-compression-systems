"""Load Qwen (or other HF causal LMs supported by tt_transformers) on TT mesh and run text generation.

Requires:
  - TT_METAL_HOME pointing at a built tt-metal tree (same as tt_transformers demos).
  - Hugging Face credentials if the checkpoint is gated (Qwen2.5-Instruct is usually public).
  - Run with the tt-metal Python venv so ttnn / transformers versions match tt-metal.

Bring-up follows ``models/tt_transformers/demo/simple_text_demo.py`` (trace region, single CQ).
Fabric defaults on; for a **single device** you can disable it via ``fabric=False`` if init fails.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from df11_tt.utils.logging import get_logger

logger = get_logger(__name__)


def _fabric_set(
    fabric_config,
    reliability_mode=None,
    fabric_tensix_config=None,
    fabric_manager=None,
    fabric_router_config=None,
):
    import ttnn

    if not fabric_config:
        return
    if reliability_mode is None:
        reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
    if fabric_tensix_config is None:
        fabric_tensix_config = ttnn.FabricTensixConfig.DISABLED
    if fabric_manager is None:
        fabric_manager = ttnn.FabricManagerMode.DEFAULT
    if fabric_router_config is not None:
        ttnn.set_fabric_config(
            fabric_config,
            reliability_mode,
            None,
            fabric_tensix_config,
            ttnn.FabricUDMMode.DISABLED,
            fabric_manager,
            fabric_router_config,
        )
    else:
        ttnn.set_fabric_config(
            fabric_config,
            reliability_mode,
            None,
            fabric_tensix_config,
            ttnn.FabricUDMMode.DISABLED,
            fabric_manager,
        )


def _fabric_reset(fabric_config):
    import ttnn

    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _ensure_tt_metal_paths(tt_metal_home: str) -> None:
    tt_metal_home = os.path.abspath(tt_metal_home)
    os.environ.setdefault("TT_METAL_HOME", tt_metal_home)
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", tt_metal_home)
    if tt_metal_home not in sys.path:
        sys.path.insert(0, tt_metal_home)
    os.chdir(tt_metal_home)


@dataclass
class GenerationResult:
    text: str
    prompt: str
    num_decode_iterations: int


class QwenTTRunner:
    """Holds mesh + tt_transformers model and runs greedy (or sampled) generation."""

    def __init__(
        self,
        tt_metal_home: str,
        hf_model: str,
        *,
        mesh_rows: int = 1,
        mesh_cols: Optional[int] = None,
        instruct: bool = True,
        max_seq_len: int = 4096,
        max_batch_size: int = 1,
        data_parallel: int = 1,
        paged_attention: bool = True,
        page_block_size: int = 32,
        page_max_num_blocks_per_dp: int = 1024,
        use_prefetcher: bool = False,
        num_layers: Optional[int] = None,
        physical_device_ids: Optional[Sequence[int]] = None,
        fabric: bool = True,
    ):
        _ensure_tt_metal_paths(tt_metal_home)
        os.environ["HF_MODEL"] = hf_model

        import ttnn
        from tests.scripts.common import get_updated_device_params

        from models.tt_transformers.demo.simple_text_demo import (
            create_tt_page_table,
            prepare_generator_args,
        )
        from models.tt_transformers.tt.common import preprocess_inputs_prefill, sample_host
        from models.tt_transformers.tt.generator import Generator, SamplingParams
        from models.tt_transformers.tt.model_config import DecodersPrecision
        from models.tt_transformers.tt.prefetcher import is_prefetcher_supported

        self._ttnn = ttnn
        self._prepare_generator_args = prepare_generator_args
        self._create_tt_page_table = create_tt_page_table
        self._preprocess_inputs_prefill = preprocess_inputs_prefill
        self._sample_host = sample_host
        self._Generator = Generator
        self._SamplingParams = SamplingParams

        avail = ttnn.get_num_devices()
        num_devices = mesh_cols if mesh_cols is not None else avail
        num_devices = min(num_devices, avail)
        if mesh_rows != 1:
            raise ValueError("This PoC only supports mesh_rows=1 (1xN mesh). Set mesh_rows=1.")
        mesh_shape = ttnn.MeshShape(1, num_devices)

        ids: Optional[List[int]] = None
        if physical_device_ids is not None:
            ids = [int(x) for x in physical_device_ids]
            if len(ids) != num_devices:
                raise ValueError(
                    f"physical_device_ids length {len(ids)} must match mesh device count {num_devices}"
                )

        use_fabric = bool(fabric) if num_devices == 1 else True
        if num_devices > 1 and not fabric:
            logger.warning("Ignoring fabric=False for multi-device mesh; fabric stays enabled.")

        device_params = {
            "fabric_config": use_fabric,
            "trace_region_size": 50_000_000,
            "num_command_queues": 1,
        }
        updated = get_updated_device_params(device_params.copy())
        fabric_config = updated.pop("fabric_config", None)
        reliability_mode = updated.pop("reliability_mode", None)
        fabric_tensix_config = updated.pop("fabric_tensix_config", None)
        fabric_manager = updated.pop("fabric_manager", None)
        fabric_router_config = updated.pop("fabric_router_config", None)

        _fabric_set(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
        self._fabric_config = fabric_config
        open_kw = dict(updated)
        if ids is not None:
            open_kw["physical_device_ids"] = ids
        self.mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **open_kw)

        use_pf = use_prefetcher and is_prefetcher_supported(self.mesh_device.get_num_devices())
        opts = lambda ma: DecodersPrecision.performance(ma.n_layers, ma.model_name)

        page_params = {"page_block_size": page_block_size, "page_max_num_blocks_per_dp": page_max_num_blocks_per_dp}

        self.model_args, self.model, self.page_table, self.tt_kv_cache, self.tokenizer, self.processor = (
            prepare_generator_args(
                num_devices=self.mesh_device.get_num_devices(),
                data_parallel=data_parallel,
                mesh_device=self.mesh_device,
                instruct=instruct,
                global_batch_size=max_batch_size * data_parallel,
                optimizations=opts,
                max_seq_len=max_seq_len,
                page_params=page_params,
                paged_attention=paged_attention,
                num_layers=num_layers,
                use_prefetcher=use_pf,
            )
        )

        self.generator = Generator(self.model, self.model_args, self.mesh_device, processor=self.processor, tokenizer=self.tokenizer)

        self.instruct = instruct
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.paged_attention = paged_attention
        self.page_params = page_params

        # simple_text_demo: Mistral-7B host sampling only
        self._models_no_dev_sampling = {"Mistral-7B"}

        self._lock = threading.Lock()

    def close(self):
        import ttnn

        for submesh in self.mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(self.mesh_device)
        _fabric_reset(self._fabric_config)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.08,
        top_k: int = 32,
        stop_at_eos: bool = True,
    ) -> GenerationResult:
        """Single-user generation (batch internally = 1)."""
        with self._lock:
            return self._generate_unlocked(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_at_eos=stop_at_eos,
            )[0]

    def generate_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.08,
        top_k: int = 32,
        stop_at_eos: bool = True,
    ) -> List[GenerationResult]:
        if len(prompts) > self.max_batch_size:
            raise ValueError(f"Got {len(prompts)} prompts, max_batch_size={self.max_batch_size}")
        with self._lock:
            return self._generate_unlocked(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_at_eos=stop_at_eos,
            )

    def _generate_unlocked(
        self,
        input_prompts: List[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_at_eos: bool,
    ) -> List[GenerationResult]:
        ttnn = self._ttnn
        model = self.model
        model_args = self.model_args
        generator = self.generator
        tt_kv_cache = self.tt_kv_cache
        page_table = self.page_table
        tokenizer = self.tokenizer
        global_batch_size = len(input_prompts)

        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        for m_args in model_args:
            if m_args.max_context_len < self.max_seq_len:
                logger.warning(
                    f"max_seq_len {self.max_seq_len} > model max_context_len {m_args.max_context_len}; clamping"
                )
        effective_max_seq = min(self.max_seq_len, model_args[0].max_context_len)

        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = self._preprocess_inputs_prefill(
            input_prompts,
            tokenizer,
            model_args,
            self.instruct,
            max_new_tokens,
            max_prefill_len=effective_max_seq,
        )

        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        if max_new_tokens + max_encoded_prompt_len > effective_max_seq:
            raise ValueError(
                f"prompt ({max_encoded_prompt_len} tok) + max_new_tokens ({max_new_tokens}) "
                f"exceeds max_seq_len ({effective_max_seq})"
            )
        if self.paged_attention:
            bs = global_batch_size
            paged_max = (
                self.page_params["page_block_size"] * self.page_params["page_max_num_blocks_per_dp"] / max(bs, 1)
            )
            if max_new_tokens + max_encoded_prompt_len > paged_max:
                raise ValueError(f"Exceeds paged KV budget (~{int(paged_max)} tokens)")

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

        supports_dev = model[0]._supports_on_device_sampling
        if model_args[0].base_model_name in self._models_no_dev_sampling:
            supports_dev = False

        device_sampling_params = None
        if supports_dev:
            device_sampling_params = self._SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        prefill_sampling = device_sampling_params if device_sampling_params is not None else None

        generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            sampling_params=prefill_sampling,
        )
        prefill_out = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            sampling_params=prefill_sampling,
        )
        if prefill_sampling is not None:
            prefilled_token, _ = prefill_out
        else:
            prefilled_token = torch.argmax(prefill_out, dim=-1)

        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(global_batch_size)]
        for user in range(global_batch_size):
            all_outputs[user].append(int(prefilled_token[user].item()))

        user_done = [False] * global_batch_size
        current_pos = torch.tensor([decoding_pos[b] for b in range(global_batch_size)])
        out_tok = prefilled_token
        iteration = 0
        users_decoding = True

        while users_decoding:
            logits, _log_probs = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=False,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                reset_batch=(iteration == 0),
                sampling_params=device_sampling_params,
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
            )

            if device_sampling_params is not None:
                out_tok = logits.unsqueeze(1)
            else:
                _, out_tok = self._sample_host(
                    logits,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    on_host=True,
                )

            current_pos = current_pos + 1
            for user in range(global_batch_size):
                user_tok = int(out_tok[user].item())
                if user_tok not in tokenizer.stop_tokens and not user_done[user]:
                    all_outputs[user].append(user_tok)
                else:
                    if stop_at_eos:
                        user_done[user] = True
                        if all(user_done):
                            users_decoding = False
                    else:
                        all_outputs[user].append(user_tok)

            iteration += 1
            if iteration >= max_new_tokens:
                users_decoding = False

        results = []
        for user, prompt in enumerate(input_prompts):
            text = tokenizer.decode(all_outputs[user])
            results.append(GenerationResult(text=text, prompt=prompt, num_decode_iterations=iteration))
        ttnn.synchronize_device(self.mesh_device)
        return results
