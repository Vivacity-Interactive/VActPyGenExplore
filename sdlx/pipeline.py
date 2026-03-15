import json, argparse, requests, torch, uuid, base64, os, sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

if __name__ == "__main__":
    common_path = sys.path.append(str(Path(__file__).resolve().parent.parent))
    if str(common_path) not in sys.path:
        sys.path.append(str(common_path))

class Settings:
    def __init__(self):
        self.config = "pipeline.json"
        self.output = "../_out/image/{uuid}.png"
        self.name = "unknown"
        self.informat = "rgb"
        self.input = ""
        self.base_path = "./"
        self.prompt = ""
        self.neg_prompt = ""
        self.num_inference_steps = 30
        self.guidance_scale = 7.5
        self.strength = 0.7
        self.seed = 12345
        self.keep_aspect = True
        self.resize_align = 64
        self.redownload = False
        self.inspect = True
        self.device_override = "cuda"
        self.mode_override = ""

class VActSDLXPipeline:
    def __init__(self):
        self.map_dtype = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp64": torch.float64,
            "i8": torch.int8,
            "i32": torch.int32,
            "i64": torch.int64,
            "b8": torch.bool,
            "cx64": torch.complex64,
            "cx128": torch.complex128
        }

    def match_resize(self, width, height, b_aspect=True, multiple=64):
        aspect = width / height if b_aspect else 1.0
        
        if width >= height:
            width = int(round(width / multiple) * multiple)
            height = int(round(width / aspect / multiple) * multiple)
        else:
            height = int(round(height / multiple) * multiple)
            width = int(round(height * aspect / multiple) * multiple)
            
        return width, height

    def try_download(self, url, dest, auth_type=None, api_key=None, api_user=None, b_redownload=False):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and not b_redownload:
            print(f"{dest} already exists, skipping download.")
            return dest
        
        print(f"downloading {url} as {dest} using auth {auth_type if auth_type else "none" } ...")

        headers = {}
        if api_key and auth_type:
            auth_method, auth_token = (auth_type.split(":", 1) + [None, None])[:2]
            if auth_method in {'basic'}:
                basic = base64(f"{api_user}:{api_key}")
                headers = { "Authorization": f"Basic {basic}" }  
            elif auth_method in {'url'}:
                url += f"&{auth_token}={api_key}"
            elif auth_method in {'header'}:
                headers = { f"{auth_token}": f"{api_key}" }
            else: # auth_method in {"bearer"}:
                headers = { "Authorization": f"Bearer {api_key}" }

        with requests.get(url, headers=headers, stream=True) as request:
            request.raise_for_status()
            total = int(request.headers.get("Content-Length", 0))
            with open(dest, "wb") as file, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
                for chunk in request.iter_content(chunk_size=8192):
                    file.write(chunk)
                    bar.update(len(chunk))

        print("download complete.")
        return dest
    
    def execute(self, settings):
        with open(settings.config) as file:
            pipeline_config = json.load(file)
        
        if not pipeline_config["type"] == "sdxl":
            print(f"{pipeline_config["type"]} is not sdxl pipeline")
            return False

        generator = torch.Generator(settings.device_override).manual_seed(settings.seed)
        
        for model_config in pipeline_config['models']:
            model_dir = Path(f"{settings.base_path}/{model_config["type"]}")
            
            auth_type = model_config.get("auth_type")
            api_key_var = model_config.get("api_key_var")
            api_user_var = model_config.get("api_user_var")
            api_key = os.getenv(api_key_var) if api_key_var else None
            api_user = os.getenv(api_user_var) if api_user_var else None

            url = model_config.get("url")
            if url:
                model_config["local_path"] = self.try_download(
                    model_config["url"], model_dir / f"{model_config['name']}.safetensors", 
                    auth_type, api_key, api_user, 
                    settings.redownload
                )
            else:
                model_config["local_path"] = model_dir / f"{model_config['name']}.safetensors"

        pipe = None
        text_encoder = None
        for model_config in pipeline_config["models"]:
            fp_mode = model_config.get("fp", "fp16")
            torch_dtype = self.map_dtype.get(fp_mode, torch.float16)
            _type = model_config["type"]
            repo_id = model_config.get("repo_id")
            if repo_id and _type in {"text_encoder"}:
                text_encoder = CLIPTextModel.from_pretrained(repo_id, token=api_key)
            elif repo_id and _type in {"controlnet"}:
                pass
            elif _type in {"checkpoint"}:
                base_model_path = model_config["local_path"]
                
                mode = settings.mode_override if settings.mode_override else model_config.get("mode", "i2t")
                mode = mode if not settings.input else "i2t"

                device = settings.device_override if settings.device_override else model_config.get("device", None)
                if mode in {'i2i'}:
                    if repo_id:
                        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_config["repo_id"], token=api_key)
                    else:
                        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                            base_model_path, 
                            torch_dtype=torch_dtype,
                            text_encoder=text_encoder
                        ).to(device)
                else:
                    if repo_id:
                        pipe = StableDiffusionXLPipeline.from_pretrained(model_config["repo_id"], token=api_key)
                    else:
                        pipe = StableDiffusionXLPipeline.from_single_file(
                            base_model_path, 
                            torch_dtype=torch_dtype,
                            text_encoder=text_encoder
                        ).to(device)
            elif _type in {"lora"}:
                if pipe is None:
                    raise RuntimeError("cannot load LoRA before any checkpoint is loaded")
                pipe.load_lora_weights(model_config["local_path"])

        if not pipe:
            raise RuntimeError("no models")
        
        image = None
        if settings.input:
            _input = f"{settings.base_path}/{settings.input}"
            image = Image.open(_input).convert(settings.informat.upper())
            if not image:
                raise RuntimeError("no imput image")
            width, height = self.match_resize(image.width, image.height, settings.keep_aspect, settings.resize_align)
            image = image.resize((width, height))
        
        image = pipe(
            prompt=settings.prompt,
            negative_prompt=settings.neg_prompt,
            image=image,
            num_inference_steps=settings.num_inference_steps, 
            guidance_scale=settings.guidance_scale,
            generator=generator,
            strength=settings.strength
        ).images[0]
        output = (settings.output
            .replace("{uuid}", uuid.uuid4().hex[:8])
            .replace("{datetime}", datetime.now().strftime("%Y%m%d_%H%M%S"))
            .replace("{name}", settings.name))
        image.save(f"{settings.base_path}/{output}")
        return True
        
    def cmd_execute(self, settings):
        parser = argparse.ArgumentParser()
        for key, value in settings.__dict__.items():
            arg_type = type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=value)
        
        args = parser.parse_args()
        
        for key in settings.__dict__.keys():
            setattr(settings, key, getattr(args, key))

        for key, value in settings.__dict__.items():
            print(f"{key}: {value}")

        while settings.inspect:
            choice = input("\ncontinue with these settings? [y/n]: ").strip().lower()
            if choice in ("y", "yes"):
                print("continuing...")
                self.execute(settings)
                return True
            elif choice in ("n", "no"):
                print("aborted by user.")
                return False
            else:
                print("Please enter 'y' or 'n'.")

if __name__ == "__main__":
    settings = Settings()
    pipeline = VActSDLXPipeline()
    pipeline.cmd_execute(settings)
