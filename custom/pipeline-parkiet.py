import torch, sys, json, os, soundfile, re, time
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
        self.output = "../_out/audio/{stem}/{id}{data}{variant}.mp3"
        self.variant = "_saskia"
        self.prompt = "[S2]"
        self.ref_audio = ""
        self.ref_text = ""
        self.skip = -1
        self.untill = 10
        self.merge_attr = 'id'
        self.text_attr = 'text'
        #self.split = '.?!\n\r'
        #self.chunks = True
        self.pause = 0.08

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

                sample_rate = processor.feature_extractor.sampling_rate
                silence = torch.zeros(
                    int(sample_rate * settings.pause),
                    dtype=torch.float32,
                    device=device
                )
                
                ref_audio = None
                ref_text = ""
                _t0 = time.perf_counter()
                if settings.ref_audio:
                    ref_audio, ref_sr = soundfile.read(settings.ref_audio.replace("{variant}",settings.variant), dtype="float32")
                    ref_text = Path(settings.ref_text).read_text(encoding="utf-8").strip() + " "

                output = self.format_output(settings.output, settings.name, settings.variant, pipe_index)
                for index, text_file in enumerate(text_files):
                    file_path = Path(text_file)
                    b_json = file_path.suffix.lower() == ".json"
                    _raw = file_path.read_text(encoding="utf-8").strip()
                    data = json.loads(_raw) if b_json else [_raw]
                    _id = data[0].get(settings.merge_attr) if data and isinstance(data[0], dict) else None
                    _merge = []
                    for _index, entry in enumerate(data):
                        b_skip = (settings.untill > 0 and _index > settings.untill) or settings.skip > _index
                        if b_skip: continue;

                        text = entry[settings.text_attr] if b_json else entry
                        id = entry[settings.merge_attr] if b_json and settings.merge_attr else None

                        inputs = processor(
                            text=(settings.prompt + " " if settings.prompt else "") + ref_text + text,# + chunk,
                            audio=ref_audio,
                            padding='max_length',
                            return_tensors="pt"
                        ).to(device)

                        prompt_len = processor.get_audio_prompt_len(
                            inputs["decoder_attention_mask"]
                        )
                        # TODO maybe sliding frame pick a part of previous entry ending
                        outputs = predictor.generate(
                            **inputs,
                            max_new_tokens=3072,
                            guidance_scale=settings.guidance,
                            temperature=1.4,
                            top_p=0.90,
                            top_k=50,
                        )
                        audio = processor.batch_decode(outputs, audio_prompt_len=prompt_len)

                        # TODO add while to merge while group attr matches finally merge and write

                        _output = output.replace("{stem}", file_path.stem).replace("{id}", f"{_index}_{id}")#.replace("{id}", id if id else settings.name)
                        if audio is not None:
                            _output_vo = Path(_output.replace("{data}", "_vo"))
                            _audio_path = settings.base_path / _output_vo
                            _audio_path.parent.mkdir(parents=True, exist_ok=True)
                            processor.save_audio(audio, _audio_path)
                            _tn = time.perf_counter()
                            print(True, len(audio), len(text), _tn - _t0, _audio_path.resolve())
                            _t0 = _tn
                           
        return True


if __name__ == "__main__":
    settings = Settings()
    pipeline = VActParkietPipeline()
    pipeline.cmd_execute(settings)