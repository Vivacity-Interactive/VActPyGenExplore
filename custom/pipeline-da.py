import torch, sys, cv2, json, os
from pathlib import Path
from depth_anything_3.api import DepthAnything3

import numpy as np

if __name__ == "__main__":
    common_path = sys.path.append(str(Path(__file__).resolve().parent.parent))
    if str(common_path) not in sys.path:
        sys.path.append(str(common_path))

from vact import VActPipelineBase, VActSettingsBase, VActPromptHandle

class Settings(VActSettingsBase):
    def __init__(self):
        super().__init__()
        self.begin = 0
        #self.output_meta = "../_out/video/{name}{variant}_meta.json"
        self.variant = ""
        self.confidence = True
        self.meta = True
        #self.process_res = 1024

class VActDAPipeline(VActPipelineBase):
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
        frame_files = [ f"{_input}/{p}" for p in os.listdir(_input) ]

        if not frame_files:
            return False

        first_frame = cv2.imread(frame_files[0])
        frame_shape = first_frame.shape[:2]
        offset = settings.begin

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
                predictor  = DepthAnything3.from_pretrained(str(model_config["repo_id"]), token=api_key).to(device)
                inference_state = predictor.inference(
                    image=frame_files,
                    #process_res=504,
                    #process_res_method="upper_bound_resize"
                )

                output = self.format_output(settings.output, settings.name, settings.variant, pipe_index)

                for frame_idx, (depth_img, conf_img) in enumerate(zip(inference_state.depth, inference_state.conf)):
                    _frame = frame_idx + offset
                    _output = output.replace("{frame}", f"{_frame:04}")
                    
                    if depth_img is not None:
                        #depth_img = (depth_data * 255).astype(np.uint8)
                        _output_depth = Path(_output.replace("{data}", "_depth"))
                        _frame_path = settings.base_path / _output_depth
                        _frame_path.parent.mkdir(parents=True, exist_ok=True)
                        b_frame_written = cv2.imwrite(str(_frame_path), depth_img)
                        print(b_frame_written, depth_img.shape, _frame_path.resolve())
                    
                    if settings.confidence and conf_img is not None:
                        #conf_img = (conf_data * 255).astype(np.uint8)
                        _output_conf = Path(_output.replace("{data}", "_conf"))
                        _frame_path = settings.base_path / _output_conf
                        _frame_path.parent.mkdir(parents=True, exist_ok=True)
                        b_frame_written = cv2.imwrite(str(_frame_path), conf_img)
                        print(b_frame_written, conf_img.shape, _frame_path.resolve())

                # output_meta = self.format_output(settings.output_meta, settings.name, settings.variant, pipe_index)
                # with open(output_meta, "w") as file:
                #     json.dump({
                #         "name": settings.name,
                #         "frame_offset": settings.begin,
                #         "frame_count": len(frame_files),
                #         "extrinsics": predictor.extrinsics,
                #         "intrinsics": predictor.intrinsics
                #     }, file)
                           
        return True


if __name__ == "__main__":
    settings = Settings()
    pipeline = VActDAPipeline()
    pipeline.cmd_execute(settings)