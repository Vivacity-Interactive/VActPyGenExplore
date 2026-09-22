import json, argparse, requests, torch, uuid, base64, os, math, cv2, time, keyboard, uuid
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
#from pynput import keyboard

# import tkinter as tk
# from tkinter import simpledialog, ttk

# from PIL import Image, ImageDraw, ImageFont

class VActSettingsBase:
    def __init__(self):
        self.config = "pipeline.json"
        self.output = "../_out/video/{name}{data}{variant}/{name}{data}{variant}_{frame}.png"
        self.name = "unknown"
        self.informat = "rgb"
        self.input = ""
        self.base_path = "./"
        self.redownload = False
        self.inspect = True
        self.device_override = "cuda"
        self.seed = 12345
        self.interactive = False

class VActPipe:
    def __init__(self, context, config, callback):
        self.next = None
        self.context = context
        self.config = config
        self.callback = callback
        self.inputs = []

    def has_next(self):
        return bool(self.next)
        
    def execute(self):
        _next =  self.next if self.next else VActPipe(self.context, None, None)
        b_next = self.callback and self.callback(self.inputs, self.context, self.config, _next.inputs)
        return _next if b_next else self

# class VActDialogHandle(simpledialog.Dialog):
#     def __init__(self, context, title = None, root = None):
#         root = root or root = tk.Tk()
#         root.withdraw()
#         super().__init__(root, title or context.__name__ or "unknown")
#         self.context = context or {}
#         self.b_pretty = True

#     def body(self, master):
#         for key, value in self.context.__dict__.items():
#             arg_type = type(value)
#             if arg_type in {list, set, dict}:
#                 arg_type = str
#                 value = json.dumps(value)
#             parser.add_argument(f"--{key}", type=arg_type, default=value)
        
#         args = parser.parse_args()
        
#         for key in settings.__dict__.keys():
#             _value = getattr(settings, key)
#             _value_type = type(_value)
#             value = getattr(args, key)
#             if _value_type in {list, set, dict}:
#                 print(value, type(value))
#                 value = json.loads(value)
#             setattr(settings, key, value)


#     def apply(self):
#         pass

class VActPipelineBase:
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

    def resolve_hue(obj_id, object_count):
        return int(255 * (obj_id % object_count) / max(object_count-1, 1))

    def model_resolve(self, model_config, model_format, config_format, settings):
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

        return model_config

    def format_input(self, input, name, variant="", pipe = -1, data="", index = -1):
        return (input
            .replace("{name}", name)
            .replace("{variant}", variant)
            .replace("{pipe}", str(pipe))
            .replace("{index}", str(index)))

    def format_output(self, output, name, variant="", pipe = -1, data="", index = -1):
        return (output
            .replace("{uuid}", uuid.uuid4().hex[:8])
            .replace("{datetime}", datetime.now().strftime("%Y%m%d_%H%M%S"))
            .replace("{name}", name)
            .replace("{variant}", variant)
            .replace("{pipe}", str(pipe))
            .replace("{index}", str(index)))

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
        return False
        
    def cmd_execute(self, settings):
        parser = argparse.ArgumentParser()
        for key, value in settings.__dict__.items():
            arg_type = type(value)
            if arg_type in {list, set, dict}:
                arg_type = str
                value = json.dumps(value)
            parser.add_argument(f"--{key}", type=arg_type, default=value)
        
        args = parser.parse_args()
        
        for key in settings.__dict__.keys():
            _value = getattr(settings, key)
            _value_type = type(_value)
            value = getattr(args, key)
            if _value_type in {list, set, dict}:
                print(value, type(value))
                value = json.loads(value)
            setattr(settings, key, value)

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

    def dialog_execute(self, settings):
        #dialog = VActDialogHandle(settings)
        pass


class VActPromptHandle:
    def __init__(self, name, frames=[], files=[], hints={}, objects=None):
        self.name = name or "Nameless"
        self.context = None
        self.frame_idx = 0
        self.object_idx = 0
        self.point_index = 0
        self.objects = objects or {"other":["other", [255,255,255]]}
        self.object_names = []
        self.hints = hints or {}
        self.frames = frames or []
        self.files = files or []
        self.masks = []
        self.b_propagate = False
        self.b_propagate_hold = False
        self.b_secondary = False
        self.b_runtime = False
        self.b_mark = False
        self.b_exit = False
        self.b_pause = True
        self.b_mask_only = False
        self.b_dirty = False
        self.b_reverse = False
        self.b_dirty_hints = True
        self.b_mouse_e = False
        self.radius = 32
        self.fps = 24
        self.blend = 0.4
        self.frame_time = 0

    def dialog_input(self, object, title):
        #dialog = VActDialogHandle(object, title)
        pass

    def remove_point(self, frame_idx, label, x, y, radius, b_mark=False):
        hints = self.hints.get(frame_idx)
        
        if hints is None: return None

        distances = [
            (ht, math.hypot(ht[1]-x, ht[2]-y)) 
            for ht in hints 
            if ht[3] is label and ht[0] is frame_idx
        ]
        near = [x for x in distances if x[1] <= radius]

        if not near:
            return None
        
        nearest = min(near, key=lambda x: x[1])[0]
        if not b_mark: hints.remove(nearest)
        else: nearest[-1] = -1
        
        return nearest

    def action_point(self, event, x, y, flags, param):
        object = self.objects[self.object_names[self.object_idx]]
        self.b_secondary = flags & cv2.EVENT_FLAG_SHIFTKEY
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.b_secondary:
                self.remove_point(self.frame_idx, 1, x, y, self.radius, self.b_mark)
            else:
                hints = self.hints.get(self.frame_idx)
                if hints is None: 
                    self.hints[self.frame_idx] = []
                    hints = self.hints[self.frame_idx]
                hints.append([self.frame_idx,x,y,1,object[0],1])
            self.b_dirty_hints = True
            if self.b_mouse_e: keyboard.send('e')
            else: self.redraw(self.frames[self.frame_idx], object, Path(self.files[self.frame_idx]))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.b_secondary:
                self.remove_point(self.frame_idx, 0, x, y, self.radius, self.b_mark)
            else:
                hints = self.hints.get(self.frame_idx)
                if hints is None: 
                    self.hints[self.frame_idx] = []
                    hints = self.hints[self.frame_idx]
                hints.append([self.frame_idx,x,y,0,object[0],1])
            self.b_dirty_hints = True
            if self.b_mouse_e: keyboard.send('e')
            else: self.redraw(self.frames[self.frame_idx], object, Path(self.files[self.frame_idx]))

    def display_info(self, frame, frame_idx, file, object, object_idx, fps):
        label_info_0 = f"frame_index({len(self.frames)}): {frame_idx:04}, object({len(self.object_names)}): ({object_idx:02}, {object[0]},"
        label_info = f"{label_info_0}   ), play_fps({self.fps*(1 - 2*self.b_reverse):02}), {"propagate" if self.b_propagate_hold else "observe"}/{"paused" if self.b_pause else "playing"}, blend({self.blend:.2f}), file: {file.resolve()}"
        
        text_pt = 0.5
        text_th = 1
        text_ot = 10
        text_cl = (255,255,255)
        circle_sz = 8

        (space_w, _), _ = cv2.getTextSize(" ", cv2.FONT_HERSHEY_DUPLEX, text_pt, text_th)
        (head_w, _), _ = cv2.getTextSize(label_info_0, cv2.FONT_HERSHEY_DUPLEX, text_pt, text_th)
        (text_w, text_h), text_bl = cv2.getTextSize(label_info, cv2.FONT_HERSHEY_DUPLEX, text_pt, text_th)
        frame = cv2.putText(frame, label_info, (text_ot, text_ot + text_h), cv2.FONT_HERSHEY_DUPLEX, text_pt, text_cl, text_th, cv2.LINE_8)
        
        color_pos = (text_ot + head_w + int(space_w * 1.5), int(text_ot + text_h/2))
        cv2.circle(frame, color_pos, circle_sz, object[1], -1)
        cv2.circle(frame, color_pos, circle_sz, text_cl, 1)
        self.b_dirty = True
        return frame

    def display_mask(self, frame, mask, alpha = 0.5):
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask
        frame = cv2.addWeighted(mask_rgb, alpha, frame, 1-alpha, 0)
        return frame

    def redraw(self, frame, object, file, mask = None):
        print("redraw")
        if frame is None: 
            return
        
        _frame = frame.copy()
        b_blend = mask is not None and not math.isclose(self.blend, 0.0, abs_tol=1e-6)
        if b_blend:
            _frame = self.display_mask(_frame, mask, self.blend)
        
        if self.b_mask_only and mask is not None: _frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask.copy()

        hints = self.hints.get(self.frame_idx) or []

        [
            cv2.circle(_frame, (ht[1],ht[2]), 5, self.objects[ht[4]][1], -1 if ht[3] else 1)
            for ht in hints
        ]
         
        _frame = self.display_info(_frame, self.frame_idx, file, object, self.object_idx, self.fps)
        cv2.imshow(self.name, _frame)
        self.b_dirty = False

    def open(self):
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.action_point)

        # no support for extending
        max_len = max(len(self.frames), len(self.files))
        self.frames.extend([None]*(max_len - len(self.frames)))
        self.files.extend([None]*(max_len - len(self.files)))
        self.masks.extend([None]*(max_len - len(self.masks)))
        self.object_names = list(self.objects.keys())

        print(self.hints)

        self.b_exit = len(self.frames) <= 0
        self.b_dirty = True

    def close(self):
        self.b_exit = True
        b_closed = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1
        if not b_closed: cv2.destroyWindow(self.name) 

    def poll(self):
        time_0 = time.time()
        #b_secondary_last = self.b_secondary
        self.b_propagate = self.b_propagate_hold
        self.frame_idx = self.frame_idx % len(self.frames)
        self.object_idx = self.object_idx % len(self.object_names)
        
        object = self.objects[self.object_names[self.object_idx]]
        frame = self.frames[self.frame_idx]
        mask = self.masks[self.frame_idx]
        file = Path(self.files[self.frame_idx])
        
        b_load_frame = frame is None and file is not None
        if b_load_frame:
            frame = self.frames[self.frame_idx] = cv2.imread(file.resolve(), cv2.IMREAD_COLOR)
            self.b_dirty = True

        if frame is None:
            self.b_exit = True
            return False
        
        if self.b_dirty:
            self.redraw(frame, object, file, mask)
            self.b_dirty = False
    
        key = cv2.waitKeyEx(int(not self.b_pause * 10))
        self.b_secondary = int(keyboard.is_pressed('shift'))
        #self.b_dirty = b_secondary_last is not self.b_secondary
        #print(f"key({key}, {self.b_secondary})")

        self.frame_time -= time.time() - time_0
        if not self.b_runtime and not self.b_pause and self.frame_time < 0:
            self.frame_time = 1/self.fps
            self.frame_idx += 1 - 2*self.b_reverse
            self.b_dirty = True
        elif self.b_runtime:
            self.frame_idx += 1 - 2*self.b_reverse
            self.b_dirty = True

        self.b_exit = (self.b_exit or key == 27
            or cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1)
        
        if not self.b_exit:
            if key == 2555904 or key == ord('d') or key == ord('D'): #right
                self.b_propagate = self.b_secondary
                self.frame_idx += 1
                self.b_dirty = True
            elif key == 2424832 or key == ord('a') or key == ord('A'): #left
                self.b_propagate = self.b_secondary
                self.frame_idx -= 1 
                self.b_dirty = True
            elif key == 2490368 or key == ord('w') or key == ord('W'): #up
                self.object_idx += 1
                self.b_dirty = True
            elif key == 2621440 or key == ord('s') or key == ord('S'): #down
                self.object_idx -= 1
                self.b_dirty = True
            elif key == 13: #enter
                if not self.b_secondary:
                    self.b_propagate_hold = not self.b_propagate_hold
                #self.b_propagate = self.b_secondary
                    self.b_dirty = True
            elif key == 32: #space
                self.b_pause = not self.b_pause
                self.b_runtime = not self.b_pause and self.b_secondary
                self.b_dirty = True
            elif key == ord('['):
                self.fps = max(1, self.fps - 1)
                self.b_dirty = True
            elif key == ord(']'):
                self.fps += 1
                self.b_dirty = True
            elif key == ord('q'):
                self.b_reverse = not self.b_reverse
                self.b_dirty = True
            elif key == ord('e') or key == ord('E'):
                if self.b_secondary:
                    self.b_mouse_e = not self.b_mouse_e
                self.b_propagate = self.b_secondary
                self.b_dirty = True
            elif key == ord(',') or key == ord('<'):
                self.blend = 0.0 if self.b_secondary else round(max(0.0, self.blend - .01), 2)
                self.b_dirty = True
            elif key == ord('.') or key == ord('>'):
                self.blend = 1.0 if self.b_secondary else round(min(1.0, self.blend + .01), 2)
                self.b_dirty = True
            elif key >= ord('0') and key <= ord('9'):
                self.object_idx = key - ord('0')
                self.b_dirty = True
    
        return not self.b_exit
    
    def get_hints(self, frame_idx=None):
        return self.hints.get(frame_idx or self.frame_idx)

    def get_hints_per_object_all(self):
        grouped = {}
        for frame_idx in self.hints.keys():
            grouped = self.get_hints_per_object(frame_idx, grouped)
        return grouped

    def get_hints_per_object(self, frame_idx=None, grouped = {}):
        hints = self.hints.get(frame_idx or self.frame_idx)
        if hints is None: return None
        grouped = grouped or {}
        for hint in hints:
            # [(<0:frame>,<1:x>,<2:y>,<3:label>,<4:name>,<5:flag>)]
            # to {<name>:[(<0:frame>,<1:points>,<2:labels>)]}
            group = grouped.get(hint[4])
            if group is None:
                grouped[hint[4]] = [-1, [], []]
                group = grouped[hint[4]]
            group[0] = hint[0]
            group[1].append([hint[1], hint[2]])
            group[2].append(hint[3])
        return grouped

    def set_objects_from_list(self, objects, b_clear=False):
        if b_clear: self.objects = {}
        for object in objects:
            self.objects[object[1]] = (object[1], object[0])
        self.object_names = list(self.objects.keys())
        self.object_idx = self.object_idx % len(self.object_names)