import torch, sys, json, os, soundfile, re
from pathlib import Path
from transformers import AutoProcessor, DiaForConditionalGeneration
import numpy as np
from array import array

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
        self.guidance = 3.0
        self.output = "../_out/audio/{name}/{name}_{stem}{data}{variant}.mp3"
        self.variant = "_saskia"
        self.prompt = "[S2]"
        self.ref_audio = ""
        self.ref_text = ""
        #self.split = '.?!\n\r'
        #self.chunks = True
        #self.pause = 0.08

class VActParkietPipeline(VActPipelineBase):
    def __init__(self):
        super().__init__()

    def execute(self, settings):
        with open(settings.config) as file:
            pipeline_config = json.load(file)

        if not pipeline_config["type"] == "custom":
            print(f"{pipeline_config["type"]} is not custom pipeline")
            return False

        ## move to base pipeline
        # for model_config in pipeline_config['models']:
        #     repro = model_config.get("repro")
        #     if not repro:
        #         self.model_resolve(model_config, "pt", "yaml", "ascii", settings)
        _input = Path(self.format_input(settings.input, settings.name, settings.variant))
        print(_input.exists(),_input.is_dir(),_input.resolve())
        text_files = [ f"{_input}/{p}" for p in os.listdir(_input) ]

        if not text_files:
            return False

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
                processor = AutoProcessor.from_pretrained(str(model_config["repo_id"]), trust_remote_code=True, token=api_key)
                predictor  = DiaForConditionalGeneration.from_pretrained(str(model_config["repo_id"]), trust_remote_code=True, torch_dtype=torch_dtype, token=api_key).to(device=device, dtype=torch_dtype)

                # sample_rate = processor.feature_extractor.sampling_rate
                # silence = torch.zeros(
                #     int(sample_rate * settings.pause),
                #     dtype=torch.float32,
                #     device=device
                # )
                # regex = re.compile(f'("[^"]*")|[{re.escape(settings.split)}]') if settings.split else r'("[^"]*")'
                ref_audio = None
                ref_text = ""
                if settings.ref_audio:
                    ref_audio, ref_sr = soundfile.read(settings.ref_audio.replace("{variant}",settings.variant), dtype="float32")
                    ref_text = Path(settings.ref_text).read_text(encoding="utf-8").strip() + " "

                output = self.format_output(settings.output, settings.name, settings.variant, pipe_index)
                for index, text_file in enumerate(text_files):
                    file_path = Path(text_file)
                    text = file_path.read_text(encoding="utf-8").strip()

                    # chunks = [
                    #     # (chunk.strip(), chunk[0] == '"')
                    #     chunk.strip()
                    #     for chunk in regex.split(text)
                    #     if chunk and chunk.strip()
                    # ] if settings.chunks else [text] #[(text, text[0] == '"')]
                    # parts = []
                    
                    # for chunk in chunks:
                    inputs = processor(
                        text=(settings.prompt + " " if settings.prompt else "") + ref_text + text,# + chunk,
                        audio=ref_audio,
                        padding='max_length',
                        return_tensors="pt"
                    ).to(device)

                    prompt_len = processor.get_audio_prompt_len(
                        inputs["decoder_attention_mask"]
                    )

                    outputs = predictor.generate(
                        **inputs,
                        max_new_tokens=3072,
                        guidance_scale=settings.guidance,
                        temperature=1.4,
                        top_p=0.90,
                        top_k=50,
                    )

                    audio = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
                        #parts.append(_audio[0])

                    #audio = torch.cat(parts, dim=0)
                    _output = output.replace("{stem}", file_path.stem)

                    if audio is not None:
                        _output_vo = Path(_output.replace("{data}", "_vo"))
                        _audio_path = settings.base_path / _output_vo
                        _audio_path.parent.mkdir(parents=True, exist_ok=True)
                        processor.save_audio(audio, _audio_path)
                        print(True, len(audio), _audio_path.resolve())
                           
        return True


if __name__ == "__main__":
    settings = Settings()
    pipeline = VActParkietPipeline()
    pipeline.cmd_execute(settings)