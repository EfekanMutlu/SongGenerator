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
import torch
import torchaudio
from typing import Optional, Literal
import register_resolvers
from omegaconf import OmegaConf
from third_party.demucs.models.pretrained import get_model_from_yaml
from codeclm.models import CodecLM
from codeclm.trainer.codec_song_pl import CodecLM_PL
from pipeline_registry import PipelineRegistry
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

class SongGenerationPipeline(PipelineRegistry):
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
    def from_pretrained(cls, ckpt_dir: str, use_accelerate: bool = True, **overrides):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
        cfg.mode = "inference"
        override_info = {
            'max_dur': ('max_dur', int),
            'min_dur': ('min_dur', int),
            'prompt_len': ('prompt_len', int),
            'qwtokenizer_max_len': ('conditioners.description.QwTokenizer.max_len', int),
            'qwtexttokenizer_max_len': ('conditioners.type_info.QwTextTokenizer.max_len', int),
        }
        for key, value in overrides.items():
            if key in override_info:
                config_path, typ = override_info[key]
                if not isinstance(value, typ):
                    raise TypeError(f"Override '{key}' must be {typ.__name__}")
                sub = cfg
                for p in config_path.split('.')[:-1]:
                    sub = getattr(sub, p)
                setattr(sub, config_path.split('.')[-1], value)
            else:
                raise ValueError(f"Unknown override '{key}'")
            
        if use_accelerate:
            with init_empty_weights():
                lite = CodecLM_PL(cfg, os.path.join(ckpt_dir, "model.pt"))
            model = load_checkpoint_and_dispatch(
                lite.audiolm, ckpt_dir, device_map="auto", offload_folder=ckpt_dir + "/offload"
            )

            model = CodecLM(
                name="from_accel",
                lm=model,
                audiotokenizer=lite.audio_tokenizer,
                max_duration=cfg.max_dur,
                seperate_tokenizer=lite.seperate_tokenizer,
            )
        else:
            lite = CodecLM_PL(cfg, os.path.join(ckpt_dir, "model.pt"))
            lite = lite.eval().to(device)
            model = CodecLM(
                name="tmp",
                lm=lite.audiolm,
                audiotokenizer=lite.audio_tokenizer,
                max_duration=cfg.max_dur,
                seperate_tokenizer=lite.seperate_tokenizer,
            )

        dm_cfg = os.path.join(ckpt_dir, "separator.yaml")
        dm_ckpt = os.path.join(ckpt_dir, "separator_model.pth")
        separator = get_model_from_yaml(dm_cfg, dm_ckpt)
        separator = separator.to(device).eval()

        auto_prompt = torch.load(os.path.join(ckpt_dir, "prompt.pt"), map_location="cpu")
        sample_rate = cfg.get("sample_rate", 48000)

        pipe = cls(cfg=cfg, sample_rate=sample_rate, device=device)
        pipe.register_modules(model=model, separator=separator, auto_prompt=auto_prompt)
        pipe.merge_prompt = [item for v in auto_prompt.values() for item in v]
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
            if auto_prompt_audio_type != "Auto" and auto_prompt_audio_type not in self.auto_prompt:
                raise ValueError(f"{auto_prompt_audio_type} geçerli değil, auto_prompt içinde yok")
            if auto_prompt_audio_type == "Auto":
                prompt_token = self.merge_prompt[
                    np.random.randint(0, len(self.merge_prompt))
                ] #TODO
            else:
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
            tokens = self.model.generate(**generate_inp, return_tokens=True) #TODO

        if melody_is_wav:
            wav_seperate = self.model.generate_audio( #TODO
                tokens, pmt_wav, vocal_wav, bgm_wav
            )
        else:
            wav_seperate = self.model.generate_audio(tokens) #TODO
       
        if output_type == "wav":
            return wav_seperate[0].cpu().float()
        elif output_type == "tensor":
            return (wav_seperate[0].cpu().float(), tokens) if return_tokens else wav_seperate[0].cpu().float()
        else:
            raise ValueError(f"output_type {output_type} desteklenmiyor.")

    def pipe(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)