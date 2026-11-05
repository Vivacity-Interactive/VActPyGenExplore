import torch, sys, json, os, soundfile, time, re
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, r"C:\_Gen\_build\qwen3-tts\build\_packages")
import transformers
print("Transformers:", transformers.__version__)
print("Path:", transformers.__file__)

from huggingface_hub import snapshot_download

import signal

if not hasattr(signal, "SIGALRM"):
    signal.SIGALRM = signal.SIGABRT
if not hasattr(signal, "alarm"):
    signal.alarm = lambda seconds: None

if __name__ == "__main__":
    common_path = sys.path.append(str(Path(__file__).resolve().parent.parent))
    if str(common_path) not in sys.path:
        sys.path.append(str(common_path))

from vact import VActPipelineBase, VActSettingsBase

class Settings(VActSettingsBase):
    def __init__(self):
        super().__init__()
        self.output = "../_out/audio/sawt/{name}/{data}/{name}{data}{variant}"
        self.variant = "_saskia"
        self.skip = -1
        self.untill = -1
        #self.regex = ""

class VActSawt4Pipeline(VActPipelineBase):
    def __init__(self):
        super().__init__()

    def execute(self, settings):
        with open(settings.config) as file:
            pipeline_config = json.load(file)

        if not pipeline_config["type"] == "custom":
            print(f"{pipeline_config["type"]} is not custom pipeline")
            return False

        _argv = sys.argv

        ## move to base pipeline
        # for model_config in pipeline_config['models']:
        #     repro = model_config.get("repro")
        #     if not repro:
        #         self.model_resolve(model_config, "pt", "yaml", "ascii", settings)
        _input = Path(self.format_input(settings.input, settings.name, settings.variant))
        print(_input.exists(),_input.is_dir(),_input.resolve())
        #regex = re.compile(settings.regex) if settings.regex else None
        #audio_files = [ f"{_input}/{p}" for p in os.listdir(_input) if not regex or regex.match(p) ]

        #if not audio_files:
        #    return False

        ## pipe = None
        pipe_index = -1
        for model_config in pipeline_config["models"]:
            pipe_index += 1
            fp_mode = model_config.get("fp", "fp16")
            torch_dtype = self.map_dtype.get(fp_mode, torch.float16)
            _type = model_config["type"]
            device_type = settings.device_override if settings.device_override else model_config.get("device", None)

            
            if _type in {"checkpoint"}:
                api_key_var = model_config.get("api_key_var")
                api_key = os.getenv(api_key_var) if api_key_var else None
                device = torch.device(device_type)
                model_path = self.get_model_path(model_config, settings, True)
                snapshot_download(model_config["repo_id"], local_dir=model_path, allow_patterns=model_config["snap_shot"], token=api_key)
                
                from sawt2.restore_v4r_fast import main as predictor
                _argv = sys.argv
                _t0 = t0 = time.perf_counter()
                
                output = self.format_output(settings.output, settings.name, settings.variant, pipe_index)
                _output = Path(output.replace("{data}", "_restored"))
                print(_output.exists(),_output.is_dir(),_output.resolve())

                #return

                sys.argv = [
                    "restore_v4r_fast.py",
                    "--ckpt", str(model_path / "generator.pt"),
                    "--vae", f"dacvae@{model_path / 'vae48_dacvae_d2.pt'}",
                    "--anchor", str(model_path / "anchor.pt"),
                    "--in", str(_input),
                    "--out", str(_output),
                    "--half",
                    "--half-anchor",
                ]

                predictor()
                
                print(_t0 - t0)
                sys.argv = _argv
        return True


if __name__ == "__main__":
    settings = Settings()
    pipeline = VActSawt4Pipeline()
    pipeline.cmd_execute(settings)