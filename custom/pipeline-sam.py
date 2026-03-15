import torch, sys, cv2, json, os, math
from pathlib import Path
from tqdm import tqdm
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as np

if __name__ == "__main__":
    common_path = sys.path.append(str(Path(__file__).resolve().parent.parent))
    if str(common_path) not in sys.path:
        sys.path.append(str(common_path))

from vact import VActPipelineBase, VActSettingsBase, VActPromptHandle

class Settings(VActSettingsBase):
    def __init__(self):
        super().__init__()
        self.objects = []
        self.begin = 0
        self.segment = "mask" # mask|color
        self.output_hints = "../_out/video/{name}{variant}_hints.json"
        self.input_hints = "../_in/video/{name}{variant}_hints.json"
        self.load_hints = True
        self.variant = ""

class VActSAMPipeline(VActPipelineBase):
    def __init__(self):
        super().__init__()

    def resolve_hue(obj_id, object_count):
        return int(255 * (obj_id % object_count) / max(object_count-1, 1))

    def model_resolve(self, model_config, model_format, config_format, trace_format, settings):
        model_dir = Path(f"{settings.base_path}/{model_config["type"]}")

        auth_type = model_config.get("auth_type")
        api_key_var = model_config.get("api_key_var")
        api_user_var = model_config.get("api_user_var")
        api_key = os.getenv(api_key_var) if api_key_var else None
        api_user = os.getenv(api_user_var) if api_user_var else None

        url = model_config.get("url")

        if url:
            model_config["local_path"] = self.try_download(
                model_config["url"], model_dir / f"{model_config['name']}.{model_format}",
                auth_type, api_key, api_user,
                settings.redownload
            )
            url_config = model_config.get("url_config")
            if url_config:
                model_config["config_path"] = self.try_download(
                    model_config["url"], model_dir / f"{model_config['name']}.{config_format}",
                    auth_type, api_key, api_user,
                    settings.redownload
                )
        else:
            model_config["local_path"] = model_dir / f"{model_config['name']}.{model_format}"
            model_config["config_path"] = model_dir / f"{model_config['name']}.{config_format}"
            #model_config["trace_path"] = model_dir / f"{model_config['name']}.{trace_format}"

        return model_config

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

        #settings.output_name = settings.output_name or settings.name
        _input = Path(self.format_input(settings.input, settings.name, settings.variant))
        _input_hints = Path(self.format_input(settings.input_hints, settings.name, settings.variant))

        hint_config = {}
        if settings.load_hints and _input_hints.is_file():
            print(_input_hints)
            with open(_input_hints) as file:
                hint_config = json.load(file)
        
        frame_files = [ f"{_input}/{p}" for p in os.listdir(_input) ]

        if not frame_files:
            return False

        first_frame = cv2.imread(frame_files[0])
        frame_shape = first_frame.shape[:2]
        b_seg_mask = settings.segment in {'mask'}
        b_seg_color = settings.segment in {'color'}

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
                predictor = SAM2VideoPredictor.from_pretrained(str(model_config["repo_id"]), token=api_key)

                with torch.inference_mode(), torch.autocast(device_type, dtype=torch_dtype):
                    inference_state = predictor.init_state(str(_input.resolve()))

                    #read model_config["trace_path"]
                    video_segments = {}
                    offset = settings.begin

                    _objects = hint_config.get("objects")
                    _hints = {int(k): v for k, v in (hint_config.get("hints") or {}).items()}
                    handle = VActPromptHandle(
                        name=settings.name, 
                        files=hint_config.get("files") or frame_files,
                        objects=_objects,
                        hints=_hints
                    )

                    if not _objects and len(settings.objects) > 0:
                        handle.set_objects_from_list(settings.objects)
                    
                    handle.open()
                    # hints = handle.get_hints_per_object_all()
                    # print(hints)
                    # if hints is not None:
                    #     predictor.reset_state(inference_state)
                    #     for obj_id, hint in hints.items():
                    #         # {<name>:[(<0:frame>,<1:points>,<2:labels>)]}
                    #         _, _objects, _masks = predictor.add_new_points_or_box(
                    #                 inference_state=inference_state,
                    #                 frame_idx=hint[0],
                    #                 obj_id=obj_id,
                    #                 points=np.array(hint[1], dtype=np.float32),
                    #                 labels=np.array(hint[2], dtype=np.int32)
                    #         )
                    
                    while handle.poll():
                        if handle.b_dirty_hints and len(handle.hints) > 0:
                            hints = handle.get_hints_per_object()
                            if hints is None: continue
                            predictor.reset_state(inference_state)
                            for obj_id, hint in hints.items():
                                # {<name>:[(<0:frame>,<1:points>,<2:labels>)]}
                                _, _objects, _masks = predictor.add_new_points_or_box(
                                        inference_state=inference_state,
                                        frame_idx=hint[0],
                                        obj_id=obj_id,
                                        points=np.array(hint[1], dtype=np.float32),
                                        labels=np.array(hint[2], dtype=np.int32)
                                    )

                        handle.b_dirty_hints = False

                        if not handle.b_propagate or len(handle.hints) <= 0:
                            continue

                        for frame_id, obj_ids, masks in predictor.propagate_in_video(inference_state):
                            video_segments[frame_id] = {
                                obj_id: (masks[i] > 0.0).cpu().numpy()
                                for i, obj_id in enumerate(obj_ids)
                            }


                        if math.isclose(handle.blend, 0.0, abs_tol=1e-6):
                            continue

                        if b_seg_color:
                            for frame_idx in sorted(video_segments.keys()):
                                mask_img = handle.masks[frame_idx]
                                if mask_img is None:
                                    mask_img = np.zeros((*frame_shape[:2], 4), dtype=np.uint8)
                                    handle.masks[frame_idx] = mask_img
                                mask_img.fill(0)
                            
                            for obj_id, mask in video_segments[frame_idx].items():
                                _object = handle.objects[obj_id]
                                mask_idxs = mask.squeeze() > 0
                                mask_np = mask_idxs * 255
                                hue = _object[1] # self.resolve_hue(obj_id, object_count)
                                mask_img[mask_idxs, :3] = hue
                                mask_img[mask_idxs, 3] = mask_np[mask_idxs]

                        if b_seg_mask:
                            for frame_idx in sorted(video_segments.keys()):
                                mask_img = handle.masks[frame_idx]
                                if mask_img is None:
                                    mask_img = np.zeros(frame_shape[:2], dtype=np.uint8)
                                    handle.masks[frame_idx] = mask_img
                                mask_img.fill(0)
                                for obj_id, mask in video_segments[frame_idx].items():
                                    mask_idxs = mask.squeeze() > 0
                                    mask_np = mask_idxs.astype(np.uint8) * 255
                                    mask_img[mask_idxs] = mask_np[mask_idxs]


                    handle.close()
                    output = self.format_output(settings.output, settings.name, settings.variant, pipe_index).replace("{data}", "_mask")
                    output_hints = self.format_output(settings.output_hints, settings.name, settings.variant, pipe_index)

                    with open(output_hints, "w") as file:
                        json.dump({
                            "name": settings.name,
                            "frame_offset": settings.begin,
                            "frame_count": len(handle.frames),
                            "objects": handle.objects,
                            "hints": handle.hints
                        }, file)

                    #print(video_segments.keys())
                    for frame_idx, mask_img in enumerate(handle.masks):
                        if mask_img is None:
                            continue
                        
                        _frame = frame_idx + offset
                        _output = Path(output.replace("{frame}", f"{_frame:04}"))
                        _frame_path = settings.base_path / _output
                        _frame_path.parent.mkdir(parents=True, exist_ok=True)
                        b_frame_written = cv2.imwrite(str(_frame_path), mask_img)
                        print(b_frame_written, mask_img.shape, _frame_path.resolve())
        return True


if __name__ == "__main__":
    settings = Settings()
    pipeline = VActSAMPipeline()
    pipeline.cmd_execute(settings)