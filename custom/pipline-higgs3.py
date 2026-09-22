import torch, sys, json, os, soundfile
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.output = "../_out/audio/{name}/{name}_{stem}{data}{variant}.mp3"
        self.variant = ""
        #self.prompt = "Speak naturally and expressively. Vary pitch and rhythm. Emphasize important words. Use a rising intonation for the question and a falling intonation for the final statement."
        self.prompt = "<|prosody:expressive_high|><|prosody:pitch_high|><|emotion:enthusiasm|>"
        #self.prompt = "Een vrouw, spreek natuurlijk en expressief. Varieer in toonhoogte en ritme. Benadruk belangrijke woorden. Gebruik een stijgende intonatie bij de vraag en een dalende intonatie bij de afsluitende mededeling."

class VActHiggs3Pipeline(VActPipelineBase):
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
                tokenizer = AutoTokenizer.from_pretrained(str(model_config["repo_id"]), trust_remote_code=True, token=api_key)
                predictor  = AutoModelForCausalLM.from_pretrained(str(model_config["repo_id"]), trust_remote_code=True, token=api_key).to(device)

                output = self.format_output(settings.output, settings.name, settings.variant, pipe_index)

                for index, text_file in enumerate(text_files):
                    file_path = Path(text_file)
                    text = file_path.read_text(encoding="utf-8").strip()
                    audio = predictor.generate_speech(
                        #text,
                        (settings.prompt if settings.prompt else "") + text,
                        tokenizer,
                        #reference_text=settings.prompt
                    )

                    _output = output.replace("{stem}", file_path.stem)

                    if audio is not None:
                        _output_vo = Path(_output.replace("{data}", "_vo"))
                        _audio_path = settings.base_path / _output_vo
                        _audio_path.parent.mkdir(parents=True, exist_ok=True)
                        soundfile.write(_audio_path, audio.cpu().numpy(), predictor.config.sample_rate)
                        print(True, audio.shape, _audio_path.resolve())
                           
        return True


if __name__ == "__main__":
    settings = Settings()
    pipeline = VActHiggs3Pipeline()
    pipeline.cmd_execute(settings)