import torch, sys, json, os, soundfile, time, re, base64, io
import numpy as np

from pathlib import Path
from google import genai

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
        self.temperature = 0.7
        self.output = "../_out/audio/{stem}/gemini/{id}{data}{variant}.mp3"
        self.variant = "_saskia"
        self.prompt = "warm and conversational"
        self.ref_audio = ""
        self.ref_consent = ""
        self.skip = -1
        self.untill = -1
        self.merge_attr = 'id'
        self.text_attr = 'text'
        self.split = '\n\r'
        self.pause = 0.08
        self.merge = True
        self.continuity = True
        self.sliding = False
        self.rate = 24000
        self.voice_key = "Kore"
        self.model = "gemini-3.8-flash-tts"
        self.store_voice = False
        self.store_data = False
        self.trim_start = 384
        self.trim_end = -3384

class VActGeminiPipeline(VActPipelineBase):
    def __init__(self):
        super().__init__()

    # def wav_as_b64(self, filename, sr_expected):
    #     data, sr = soundfile.read(filename, dtype="int16")
    #     if sr != sr_expected: raise ValueError(f"{filename}: expected {sr_expected} Hz, got {sr}")
    #     if data.ndim > 1: data = data.mean(dim=1)
    #     _data = io.BytesIO()
    #     soundfile.write(_data, data, sr, format="WAV", subtype="PCM_16")
    #     return base64.b16encode(_data.getvalue()).decode("ascii"), sr

    def wav_as_b64(self, filename, sr_expected):
        with open(filename, "rb") as f: return base64.b64encode(f.read()).decode("utf-8"), sr_expected

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
                predictor = genai.Client()

                sample_rate = settings.rate
                # silence = torch.zeros(
                #     int(sample_rate * settings.pause),
                #     dtype=torch.float32,
                #     device=device
                # )

                split = re.compile(f"[{re.escape(settings.split)}]")
                
                #con_audio = None
                #con_text = None
                voice_key = settings.voice_key
                _t0 = t0 = time.perf_counter()
                if settings.ref_audio:
                    ref_audio, ref_sr = self.wav_as_b64(settings.ref_audio.replace("{variant}",settings.variant), settings.rate)
                    ref_consent, refc_sr = self.wav_as_b64(settings.ref_consent.replace("{variant}",settings.variant), settings.rate)
                    voice = predictor.voices.create(
                        store=settings.store_voice,
                        voice={
                            "model": settings.model,
                            "type": "replicated",
                            "language_code": "nl-NL",
                            "display_name": settings.variant.replace('_', " ").strip().title(),
                            "replicated": {
                                "source_audio": {
                                    "mime_type": "audio/wav",
                                    "data": ref_audio,
                                },
                                "consent_audio": {
                                    "mime_type": "audio/wav",
                                    "data": ref_consent,
                                },
                            },
                        },
                    )
                    voice_key = voice.id if settings.store_voice else voice.key
                    print(voice_key)

                output = self.format_output(settings.output, settings.name, settings.variant, pipe_index)
                for index, text_file in enumerate(text_files):
                    file_path = Path(text_file)
                    b_json = file_path.suffix.lower() == ".json"
                    _raw = file_path.read_text(encoding="utf-8").strip()
                    data = json.loads(_raw) if b_json else [_raw]
                    #_id = data[0].get(settings.merge_attr) if data and isinstance(data[0], dict) else None
                    _merge = []
                    for _index, entry in enumerate(data):
                        b_skip = (settings.untill > 0 and _index > settings.untill) or settings.skip > _index
                        if b_skip: continue;

                        text = entry[settings.text_attr] if b_json else entry
                        id = entry[settings.merge_attr] if b_json and settings.merge_attr else None

                        #chuncks = split.split(text)
                        #for index_, chunk in enumerate(chuncks):
                        
                        response = predictor.interactions.create(
                            model = settings.model,
                            input = [{
                                "type": "text",
                                "text": text,
                                "annotations": [{
                                    "type": "speech_metadata",
                                    "style": settings.prompt
                                }] if settings.prompt else None
                            }],
                            response_format={"type": "audio"},
                            generation_config={ "speech_config": [{"voice": voice_key }] },
                            store=settings.store_data
                        )

                        _audio = response.output_audio.data if response.output_audio else ""

                        #con_audio = _audio if settings.continuity else None
                        #con_text = text if settings.continuity else None

                        b_write = (not settings.merge) or (settings.merge and ( (_index + 1) >= len(data) or data[_index + 1].get(settings.merge_attr) != id))
                        if not b_write: _merge.append(_audio)
                        else:
                            _merge.append(_audio)
                            audio = _audio if not settings.merge else "".join(_merge)
                            audio = base64.b64decode(audio)
                            _merge = []
                            _output = output.replace("{stem}", file_path.stem).replace("{id}", f"{_index}_{id}")#.replace("{id}", id if id else settings.name)
                            if audio is not None:
                                _output_vo = Path(_output.replace("{data}", "_vo"))
                                _audio_path = settings.base_path / _output_vo
                                _audio_path.parent.mkdir(parents=True, exist_ok=True)
                                pcm = np.frombuffer(audio, dtype=np.int16)
                                if settings.trim_end or settings.trim_start: pcm = pcm[settings.trim_end:settings.trim_start]
                                soundfile.write(_audio_path, pcm, sample_rate, format="MP3")
                                _tn =  time.perf_counter()
                                print(True, len(audio), len(text), _tn - _t0, f"f({index}/{len(text_files)})",f"t({_index}/{len(data)})", _audio_path.resolve())
                                _t0 = _tn
                print(_t0 - t0)
        return True


if __name__ == "__main__":
    settings = Settings()
    pipeline = VActGeminiPipeline()
    pipeline.cmd_execute(settings)