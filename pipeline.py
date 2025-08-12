import os
import sys
os.environ["USER"] = "root"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "third_party/hub")
os.environ["NCCL_HOME"] = "/usr/local/tccl"

sys.path.insert(0, os.path.join(os.getcwd(), "codeclm/tokenizer/"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "codeclm/tokenizer/Flow1dVAE/"))

import json
import time
import numpy as np
import pickle
import torch
import torchaudio
from typing import Optional, Literal
import register_resolvers
from omegaconf import OmegaConf
from third_party.demucs.models.pretrained import get_model_from_yaml
from codeclm.models import CodecLM
from codeclm.trainer.codec_song_pl import CodecLM_PL
from pipeline_registry import PipelineRegistry
#from accelerate import init_empty_weights, load_checkpoint_and_dispatch

class SongGenerationPipeline(PipelineRegistry):
    _ram_cache: dict = {}
    def __init__(
        self,
        model=None,
        separator=None,
        auto_prompt=None,
        cfg=None,
        sample_rate: int = 48000,
        device: Optional[torch.device] = None,
    ):
        super().__init__() 

        self.model = None
        self.separator = None
        self.auto_prompt = None
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.merge_prompt = None

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, **overrides):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = os.path.expanduser("~/.cache/SongGenerator")
        os.makedirs(cache_dir, exist_ok=True)
        override_key = "_".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        cache_key = f"{os.path.basename(ckpt_dir)}_{override_key or 'default'}"
    
        model_pt = os.path.join(cache_dir, f"{cache_key}_model.pt")
        separator_pt = os.path.join(cache_dir, f"{cache_key}_separator.pt")
        prompt_pt = os.path.join(cache_dir, f"{cache_key}_prompt.pt")
        meta_json = os.path.join(cache_dir, f"{cache_key}_meta.json")
    
        # --- 1) RAM CACHE CONTROL ---
        if cache_key in cls._ram_cache:
            print("RAM loading from cache...")
            return cls._ram_cache[cache_key]
    
        # --- 2) build() DEFINE ---
        def build():
            cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
            cfg.mode = "inference"
            for key, value in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
            lite = CodecLM_PL(cfg, os.path.join(ckpt_dir, "model.pt"))
            lite = lite.eval().to(device)
            model = CodecLM(
                name="tmp",
                lm=lite.audiolm,
                audiotokenizer=lite.audio_tokenizer,
                max_duration=cfg.max_dur,
                seperate_tokenizer=lite.seperate_tokenizer,
            )
    
            root = os.getcwd()
            sep = get_model_from_yaml(
                os.path.join(root, "third_party/demucs/ckpt/htdemucs.yaml"),
                os.path.join(root, "third_party/demucs/ckpt/htdemucs.pth"))
            sep = sep.to(device).eval()
    
            prompts = torch.load(os.path.join(root, "ckpt/prompt.pt"), map_location="cpu")
            pipe = cls(cfg=cfg, sample_rate=cfg.get("sample_rate",48000), device=device)
            pipe.register_modules(model=model, separator=sep, auto_prompt=prompts)
            pipe.merge_prompt = [item for v in prompts.values() for item in v]
            return pipe
    
        # --- 3) DISK CACHE CONTROL ---
        if all(os.path.exists(f) for f in [model_pt, separator_pt, prompt_pt, meta_json]):
            print("Disk loading from cache...")
            meta = json.load(open(meta_json))
            pipe = build()
            pipe.model.load_state_dict(torch.load(model_pt, map_location=device))
            pipe.separator.load_state_dict(torch.load(separator_pt, map_location=device))
            pipe.auto_prompt = torch.load(prompt_pt, map_location="cpu")
            pipe.merge_prompt = [item for v in pipe.auto_prompt.values() for item in v]
            cls._ram_cache[cache_key] = pipe
            return pipe
    
        # --- 4) IF NO CACHE ---
        print("Building new pipeline...")
        pipe = build()
        torch.save(pipe.model.state_dict(), model_pt)
        torch.save(pipe.separator.state_dict(), separator_pt)
        torch.save(pipe.auto_prompt, prompt_pt)
        with open(meta_json, "w") as f:
            json.dump({"ckpt_dir": ckpt_dir, "overrides": overrides}, f)
        cls._ram_cache[cache_key] = pipe
        return pipe

    def _load_audio(self, f):
        a, fs = torchaudio.load(f)
        if fs != 48000:
            a = torchaudio.functional.resample(a, fs, 48000)
        if a.shape[-1] >= 48000 * 10:
            a = a[..., : 48000 * 10]
        else:
            a = torch.cat([a, a], -1)
        return a[:, 0 : 48000 * 10]

    def _separate_audio(self, audio_path, output_dir="tmp", ext=".wav"):
        os.makedirs(output_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []

        for stem in self.separator.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 1:
            vocal_path = output_paths[0]
        else:
            drums_path, bass_path, other_path, vocal_path = self.separator.separate(
                audio_path, output_dir, device=self.device
            )
            for path in [drums_path, bass_path, other_path]:
                os.remove(path)
        full_audio = self._load_audio(audio_path)
        vocal_audio = self._load_audio(vocal_path)
        bgm_audio = full_audio - vocal_audio
        return full_audio, vocal_audio, bgm_audio

    @torch.no_grad()
    def __call__(
        self,
        gt_lyric: str,
        descriptions: Optional[str] = None,
        prompt_audio_path: Optional[str] = None,
        auto_prompt_audio_type: Optional[str] = None,
        melody_wavs: Optional[torch.Tensor] = None,
        vocal_wavs: Optional[torch.Tensor] = None,
        bgm_wavs: Optional[torch.Tensor] = None,
        melody_is_wav: Optional[bool] = None,
        idx: Optional[str] = None,
        return_tokens: bool = False,
        output_type: Literal["wav", "tensor"] = "wav",
    ):

        # Prompt input priority: prompt_audio_path > auto_prompt_audio_type > direct wavs > None
        if prompt_audio_path and auto_prompt_audio_type:
            raise ValueError("Only one of prompt_audio_path or auto_prompt_audio_type can be used.")
    
        if prompt_audio_path:
            pmt_wav, vocal_wav, bgm_wav = self._separate_audio(prompt_audio_path)
            melody_is_wav = True
        elif auto_prompt_audio_type:
            if (
                self.auto_prompt is not None
                and auto_prompt_audio_type != "Auto"
                and auto_prompt_audio_type not in self.auto_prompt
            ):
                raise ValueError(f"{auto_prompt_audio_type} is not a valid key in auto_prompt. Available keys: {list(self.auto_prompt.keys())}")
            if auto_prompt_audio_type == "Auto":
                assert self.merge_prompt is not None
                prompt_token = self.merge_prompt[
                    np.random.randint(0, len(self.merge_prompt))
                ] #TODO
            else:
                assert self.auto_prompt is not None
                prompt_token = self.auto_prompt[auto_prompt_audio_type][
                    np.random.randint(0, len(self.auto_prompt[auto_prompt_audio_type]))
                ]
            pmt_wav = prompt_token[:, [0], :]
            vocal_wav = prompt_token[:, [1], :]
            bgm_wav = prompt_token[:, [2], :]
            melody_is_wav = False
        elif melody_wavs is not None and vocal_wavs is not None and bgm_wavs is not None:
            pmt_wav = melody_wavs
            vocal_wav = vocal_wavs
            bgm_wav = bgm_wavs
            melody_is_wav = melody_is_wav if melody_is_wav is not None else True
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = melody_is_wav if melody_is_wav is not None else True
    
        generate_inp = {
            "lyrics": [gt_lyric.replace("  ", " ")],
            "descriptions": [descriptions] if descriptions is not None else [""],
            "melody_wavs": pmt_wav,
            "vocal_wavs": vocal_wav,
            "bgm_wavs": bgm_wav,
            "melody_is_wav": melody_is_wav,
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            assert self.model is not None, "model is not registered or register_modules is not called."
            tokens = self.model.generate(**generate_inp, return_tokens=True) #TODO

        if melody_is_wav:
            assert self.model is not None, "model is not registered or register_modules is not called."
            wav_seperate = self.model.generate_audio( #TODO
                tokens, pmt_wav, vocal_wav, bgm_wav
            )
        else:
            assert self.model is not None, "model is not registered or register_modules is not called."
            wav_seperate = self.model.generate_audio(tokens) #TODO
       
        if output_type == "wav":
            return wav_seperate[0].cpu().float()
        elif output_type == "tensor":
            return (wav_seperate[0].cpu().float(), tokens) if return_tokens else wav_seperate[0].cpu().float()
        else:
            raise ValueError(f"output_type {output_type} is not supported. Use 'wav' or 'tensor'.")

    def pipe(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)