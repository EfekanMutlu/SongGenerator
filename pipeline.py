import os
import sys

######## TODO: This part should be moved to Docker
os.environ["USER"] = "root"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "third_party/hub")
os.environ["NCCL_HOME"] = "/usr/local/tccl"

sys.path.insert(0, os.path.join(os.getcwd(), "codeclm/tokenizer/"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "codeclm/tokenizer/Flow1dVAE/"))
########

import json
import time

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from third_party.demucs.models.pretrained import get_model_from_yaml

from codeclm.models import CodecLM
from codeclm.trainer.codec_song_pl import CodecLM_PL


class SongGenerationPipeline:
    def __init__(self, ckpt_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = ckpt_dir
        self._load_model()
        self._load_separator()
        self._load_prompts()

    def _load_model(self):
        OmegaConf.register_resolver("eval", lambda x: eval(x))
        OmegaConf.register_resolver(
            "concat", lambda *x: [xxx for xx in x for xxx in xx]
        )
        OmegaConf.register_resolver(
            "get_fname", lambda: "dummy"
        )  # input_jsonl yoksa dummy koyabilirsin
        OmegaConf.register_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))

        cfg_path = os.path.join(self.ckpt_dir, "config.yaml")
        ckpt_path = os.path.join(self.ckpt_dir, "model.pt")
        cfg = OmegaConf.load(cfg_path)
        cfg.mode = "inference"
        self.cfg = cfg
        self.max_duration = cfg.max_dur

        model_light = CodecLM_PL(cfg, ckpt_path)
        model_light = model_light.eval().to(self.device)
        model_light.audiolm.cfg = cfg
        self.model = CodecLM(
            name="tmp",
            lm=model_light.audiolm,
            audiotokenizer=model_light.audio_tokenizer,
            max_duration=self.max_duration,
            seperate_tokenizer=model_light.seperate_tokenizer,
        )

    def _load_separator(self):
        dm_model_path = "third_party/demucs/ckpt/htdemucs.pth"
        dm_config_path = "third_party/demucs/ckpt/htdemucs.yaml"
        model = get_model_from_yaml(dm_config_path, dm_model_path)
        model.to(self.device)
        model.eval()
        self.separator = model

    def _load_prompts(self):
        self.auto_prompt = torch.load("ckpt/prompt.pt")
        self.merge_prompt = [
            item for sublist in self.auto_prompt.values() for item in sublist
        ]

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

    def generate(self, input_jsonl, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "audios"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "jsonl"), exist_ok=True)

        with open(input_jsonl, "r") as fp:
            lines = fp.readlines()

        new_items = []
        for line in lines:
            item = json.loads(line)
            target_wav_name = os.path.join(save_dir, "audios", f"{item['idx']}.wav")
            lyric = item["gt_lyric"]
            descriptions = item.get("descriptions", None)

            if "prompt_audio_path" in item:
                assert os.path.exists(
                    item["prompt_audio_path"]
                ), f"prompt_audio_path {item['prompt_audio_path']} not found"
                assert (
                    "auto_prompt_audio_type" not in item
                ), f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
                pmt_wav, vocal_wav, bgm_wav = self._separate_audio(
                    item["prompt_audio_path"]
                )
                melody_is_wav = True
            elif "auto_prompt_audio_type" in item:
                assert (
                    item["auto_prompt_audio_type"] in self.auto_prompt
                ), f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
                if item["auto_prompt_audio_type"] == "Auto":
                    prompt_token = self.merge_prompt[
                        np.random.randint(0, len(self.merge_prompt))
                    ]
                else:
                    prompt_token = self.auto_prompt[item["auto_prompt_audio_type"]][
                        np.random.randint(
                            0, len(self.auto_prompt[item["auto_prompt_audio_type"]])
                        )
                    ]
                pmt_wav = prompt_token[:, [0], :]
                vocal_wav = prompt_token[:, [1], :]
                bgm_wav = prompt_token[:, [2], :]
                melody_is_wav = False
            else:
                pmt_wav = None
                vocal_wav = None
                bgm_wav = None
                melody_is_wav = True

            generate_inp = {
                "lyrics": [lyric.replace("  ", " ")],
                "descriptions": [descriptions],
                "melody_wavs": pmt_wav,
                "vocal_wavs": vocal_wav,
                "bgm_wavs": bgm_wav,
                "melody_is_wav": melody_is_wav,
            }
            start_time = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                tokens = self.model.generate(**generate_inp, return_tokens=True)
            mid_time = time.time()

            with torch.no_grad():
                if melody_is_wav:
                    wav_seperate = self.model.generate_audio(
                        tokens, pmt_wav, vocal_wav, bgm_wav
                    )
                else:
                    wav_seperate = self.model.generate_audio(tokens)
            end_time = time.time()
            torchaudio.save(
                target_wav_name, wav_seperate[0].cpu().float(), self.cfg.sample_rate
            )
            print(
                f"process{item['idx']}, lm cost {mid_time - start_time}s, diffusion cost {end_time - mid_time}"
            )

            item["idx"] = f"{item['idx']}"
            item["wav_path"] = target_wav_name
            new_items.append(item)

        src_jsonl_name = os.path.basename(input_jsonl)
        with open(
            os.path.join(save_dir, "jsonl", f"{src_jsonl_name}.jsonl"),
            "w",
            encoding="utf-8",
        ) as fw:
            for item in new_items:
                fw.writelines(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SongGeneration Inference Script")
    parser.add_argument("--input_jsonl", type=str, help="Input JSONL file")
    parser.add_argument(
        "--save_dir", type=str, default=os.getcwd(), help="Directory to save outputs"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./ckpt/songgeneration_base",
        help="Checkpoint directory",
    )
    args = parser.parse_args()

    pipeline = SongGenerationPipeline(ckpt_dir=args.ckpt_dir)
    pipeline.generate(input_jsonl=args.input_jsonl, save_dir=args.save_dir)
