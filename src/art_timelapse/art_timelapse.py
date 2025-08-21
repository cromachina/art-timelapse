from pathlib import Path
from zipfile import ZipFile
import argparse
import asyncio
import json
import os
import time
import tkinter as tk
import zipfile

from PIL import Image, ImageOps
from psd_tools import PSDImage
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import cv2
import numpy as np
import pynput
import pywinctl
import tqdm
from mss import mss
from . import sai

def even_dim(a):
    return a if a % 2 == 0 else a - 1

def point_in_bbox(bbox, point):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

def nth_counter(nth):
    counter = 1
    while True:
        if counter == 1:
            counter = nth
            yield True
        else:
            counter -= 1
            yield False

def get_nth_frame(config, frame_count):
    nth_frame = 1
    if config.export_time_limit > 0:
        target_frames = config.export_time_limit * float(config.fps)
        if frame_count > target_frames:
            nth_frame = np.floor(frame_count / target_frames)
    return nth_frame

def set_rect_params(rect, width, height, alpha):
    rect.overrideredirect(1)
    rect.geometry(f'{width}x{height}+0+0')
    rect.wait_visibility()
    rect.configure(bg='black')
    rect.attributes('-topmost', True)
    rect.attributes('-alpha', alpha)

def make_rect(parent, width, height):
    rect = tk.Toplevel(parent)
    set_rect_params(rect, width, height, 0.5)
    return rect

def get_spanned_rect(pos1, pos2):
    p1x, p1y = pos1
    p2x, p2y = pos2
    return min(p1x, p2x), min(p1y, p2y), max(p1x, p2x), max(p1y, p2y)

def get_bbox_from_drag_rect():
    win = tk.Tk()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    set_rect_params(win, sw, sh, 0.25)
    top_rect = make_rect(win, 0, 0)
    left_rect = make_rect(win, 0, 0)
    right_rect = make_rect(win, 0, 0)
    bottom_rect = make_rect(win, 0, 0)
    pos1 = None
    pos2 = None
    cancelled = False

    def win_motion(event):
        nonlocal pos2
        if pos1 is not None:
            pos2 = (event.x, event.y)
            lx, ly, rx, ry = get_spanned_rect(pos1, pos2)
            top_rect.geometry(f'{sw}x{ly}+0+0')
            left_rect.geometry(f'{lx}x{sh - ly}+0+{ly}')
            right_rect.geometry(f'{sw - rx}x{sh - ly}+{rx}+{ly}')
            bottom_rect.geometry(f'{rx - lx}x{sh - ry}+{lx}+{ry}')

    def set_clicked(event):
        nonlocal pos1
        win.attributes('-alpha', 0)
        pos1 = (event.x, event.y)
        win_motion(event)

    def set_released(event):
        win.destroy()

    def set_cancelled(event):
        nonlocal cancelled
        win.destroy()
        cancelled = True

    win.bind('<Motion>', win_motion)
    win.bind('<ButtonPress-1>', set_clicked)
    win.bind('<ButtonRelease-1>', set_released)
    win.bind('<ButtonPress-3>', set_cancelled)
    win.lift()
    win.mainloop()
    if cancelled:
        return None
    else:
        return *pos1, *pos2

def get_window_from_click():
    print('Click on a subwindow to start recording. Right click to cancel.')
    window = None
    def on_click(x, y, button, is_pressed):
        nonlocal window
        if is_pressed:
            if button == pynput.mouse.Button.left:
                window = pywinctl.getTopWindowAt(x, y)
                mouse_listener.stop()
            elif button == pynput.mouse.Button.right:
                mouse_listener.stop()
    mouse_listener = pynput.mouse.Listener(on_click=on_click)
    mouse_listener.start()
    mouse_listener.join()
    return window

def drag_window_and_bbox():
    print('Click and drag on a window to record that area. Right click to cancel selection.')
    bbox = get_bbox_from_drag_rect()
    if bbox is None:
        return None, None
    window = pywinctl.getTopWindowAt(*bbox[:2])
    return window, bbox

def get_window_and_bbox(config, metadata=None):
    if config.use_last_grab and metadata is not None:
        bbox = metadata.get_bbox()
        if bbox is None:
            return drag_window_and_bbox()
        else:
            return get_window_from_click(), bbox
    elif config.drag_grab:
        return drag_window_and_bbox()
    else:
        return get_window_from_click(), None

def get_window_from_pid(pid):
    result = None
    for window in pywinctl.getAllWindows():
        if window.getPID() == pid:
            result = window
    if result is None:
        print('Could not find window automatically')
        result = get_window_from_click()
    return result

class Metadata:
    def __init__(self, frames_file:ZipFile):
        self.frames_file = frames_file
        self.load()

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        try:
            self.save()
        except:
            pass

    def load(self):
        str_data = self.frames_file.comment.decode()
        try:
            metadata = json.loads(str_data)
            if isinstance(metadata, dict):
                self.metadata = metadata
            else:
                self.metadata = {}
        except:
            self.metadata = {}

    def save(self):
        self.frames_file.comment = json.dumps(self.metadata).encode()

    def get_max_size(self):
        return tuple(self.metadata.get('max_size', (0, 0)))

    def update_max_size(self, new_size):
        old_size = self.get_max_size()
        self.metadata['max_size'] = tuple(max(a, b) for a, b in zip(old_size, new_size))

    def get_bbox(self):
        bbox = self.metadata.get('bbox')
        if bbox is not None:
            bbox = tuple(bbox)
        return bbox

    def set_bbox(self, bbox):
        if bbox is None:
            return
        old_bbox = self.get_bbox()
        self.metadata['bbox'] = bbox

class VideoWriter():
    def __init__(self, file_path, fps, size):
        self.size = tuple(map(lambda x: even_dim(int(x)), size))
        self.video_writer = cv2.VideoWriter(str(file_path), cv2.VideoWriter.fourcc(*'avc1'), fps, self.size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.video_writer.release()

    def write(self, img):
        if img.size != self.size:
            img = ImageOps.pad(img, self.size)
        data = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        self.video_writer.write(data)

    def write_numpy(self, data):
        self.video_writer.write(data)

class VideoReader():
    def __init__(self, file_path):
        self.video_reader = cv2.VideoCapture(str(file_path))

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.video_reader.release()

    def read(self):
        return self.video_reader.read()

    def get_frame_count(self):
        return self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_size(self):
        return self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH), self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)

class InputTracker():
    def __init__(self, target_window, bbox=None):
        self.target_window = target_window
        self.bbox = bbox
        self.click_started_in_window = False
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_args):
        self.stop()

    def on_click_callback(self, x, y, button, is_pressed):
        if is_pressed:
            window = pywinctl.getTopWindowAt(x, y)
            self.click_started_in_window = window.getHandle() == self.target_window.getHandle()
            if self.click_started_in_window:
                bbox = window.rect if self.bbox is None else self.bbox
                self.click_started_in_window = point_in_bbox(bbox, (x, y))
        if not is_pressed and self.click_started_in_window:
            self.click_started_in_window = False
            self.loop.call_soon_threadsafe(self.queue.put_nowait, None)

    async def get_event(self):
        await self.queue.get()

    async def track_window(self):
        while self.target_window.isAlive:
            await asyncio.sleep(0.1)
        self.queue.shutdown()

    def start(self):
        self.mouse_listener = pynput.mouse.Listener(on_click=self.on_click_callback)
        self.mouse_listener.start()
        self._wait_task = asyncio.create_task(self.track_window())

    def stop(self):
        self.mouse_listener.stop()

async def get_input_tracker_events(window, bbox=None):
    with InputTracker(window, bbox) as input_tracker:
        print(f'Tracking input for window: {window.title}')
        try:
            while True:
                yield await input_tracker.get_event()
        except asyncio.CancelledError:
            print(f'Stopping tracking window: {window.title}')
        except asyncio.QueueShutDown:
            print('Window lost; Stopping now')

class FileTracker(FileSystemEventHandler):
    def __init__(self, target_file):
        self.target_file = target_file
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        super().__init__()
        self.observer = Observer()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_args):
        self.stop()

    def on_closed(self, event):
        if Path(event.src_path) == self.target_file:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, None)

    async def get_event(self):
        await self.queue.get()

    def start(self):
        self.observer.schedule(self, self.target_file.parent)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()

async def get_file_tracker_events(target_file):
    with FileTracker(target_file) as file_tracker:
        try:
            print(f'Started watching file for changes: {target_file}')
            while True:
                yield await file_tracker.get_event()
        except asyncio.CancelledError:
            print(f'Stopped watching file for changes: {target_file}')

# Export frames captured to a zip file to video.
def run_export(config):
    with ZipFile(config.frame_data, 'r') as zfile, Metadata(zfile) as metadata:
        frames = zfile.namelist()
        frames.sort()
        frame_count = len(frames)
        video_file = Path(config.frame_data).with_suffix('.mp4')
        print(f'Exporting {video_file}')
        counter = nth_counter(get_nth_frame(config, frame_count))
        with VideoWriter(video_file, config.fps, metadata.get_max_size()) as video_writer:
            for frame in tqdm.tqdm(frames, unit='frames'):
                if not next(counter):
                    continue
                with zfile.open(frame, 'r') as mfile:
                    img = Image.open(mfile)
                    video_writer.write(img)
        print('Finished exporting')

# Convert a video to a different time scale or FPS.
def run_convert_video(config):
    with VideoReader(config.video_file) as video_reader:
        frame_count = video_reader.get_frame_count()
        nth_frame = get_nth_frame(config, frame_count)
        if nth_frame > 1:
            short_name = config.video_file.with_stem(config.video_file.stem + '-short')
            print(f'Converting to {short_name}')
            counter = nth_counter(nth_frame)
            with VideoWriter(short_name, config.fps, video_reader.get_size()) as video_writer, tqdm.tqdm(total=frame_count, unit='frames') as prog:
                while True:
                    ret, frame = video_reader.read()
                    prog.update(1)
                    if ret == False:
                        break
                    if not next(counter):
                        continue
                    video_writer.write_numpy(frame)
            print('Finished converting')
        else:
            print('Input video already satisfies time constraint; no conversion performed')

def grab(sct, bbox):
    img = sct.grab(bbox)
    return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

def sai_version_check(sai_proc):
    if not sai_proc.is_sai_version_compatible():
        print('Warning: Capture may not work correctly')
        res = input('Continue? [y/N]').lower()
        return res.startswith('y')
    return True

def select_canvas(sai_proc):
    while True:
        canvas_list = sai_proc.get_canvas_list()
        print('Select a canvas to record (Ctrl+C to cancel):')
        for i, canvas in zip(range(len(canvas_list)), canvas_list):
            print(f'[{i + 1}]', canvas.get_name())
        res = input(f'Enter index [1-{len(canvas_list)}]:')
        try:
            res = int(res)
            canvas = canvas_list[res - 1]
        except ValueError:
            print('Could not parse input, trying again')
        except IndexError:
            print('Index out of range, trying again')
        else:
            return canvas

def is_different_image(a, b):
    if a is None or b is None:
        return True
    return np.any(a != b)

# Capture composite images directly from SAI.
# Allows continuing capture after stopping and variable frame size.
async def run_sai_capture_to_frames(config):
    with (ZipFile(config.frame_data, 'a', zipfile.ZIP_DEFLATED) as zfile,
            Metadata(zfile) as metadata,
            sai.SAI() as sai_proc):
        if not sai_version_check(sai_proc):
            return
        canvas = select_canvas(sai_proc)
        window = get_window_from_pid(sai_proc.get_pid())
        if window is None:
            return
        counter = nth_counter(config.nth_frame)
        last_img = None
        async for _ in get_input_tracker_events(window):
            if not next(counter):
                continue
            if not sai_proc.check_if_canvas_exists(canvas):
                print('Canvas lost, stopping now')
                return
            img = sai_proc.get_canvas_image(canvas)
            if not is_different_image(last_img, img):
                continue
            last_img = img
            img = Image.fromarray(img)
            img.thumbnail((config.image_size_limit, config.image_size_limit))
            metadata.update_max_size(img.size)
            with zfile.open(str(time.time_ns()), 'w') as mfile:
                img.save(mfile, format='jpeg', quality=95)
        print('Finished recording')

# Capture a window's rectangular subregion to a zip file to later be converted to video.
# Allows continuing capture after stopping and variable frame size.
async def run_capture_to_frames(config):
    with ZipFile(config.frame_data, 'a', zipfile.ZIP_DEFLATED) as zfile, Metadata(zfile) as metadata, mss() as sct:
        window, bbox = get_window_and_bbox(config, metadata)
        if window is None:
            return
        metadata.set_bbox(bbox)
        counter = nth_counter(config.nth_frame)
        print(f'Frame data: {config.frame_data}')
        async for _ in get_input_tracker_events(window, bbox):
            if not next(counter):
                continue
            img = grab(sct, bbox)
            metadata.update_max_size(img.size)
            with zfile.open(str(time.time_ns()), 'w') as mfile:
                img.save(mfile, format='jpeg', quality=95)
        print('Finished recording')

# Capture a window's rectangular subregion directly to a video file.
async def run_capture_to_video(config):
    window, bbox = get_window_and_bbox(config)
    if window is None:
        return
    if bbox is None:
        bbox = window.bbox
    with VideoWriter(config.video_file, config.fps, (bbox[2] - bbox[0], bbox[3] - bbox[1])) as video_writer, mss() as sct:
        counter = nth_counter(config.nth_frame)
        print(f'Video output: {config.video_file}')
        async for _ in get_input_tracker_events(window, bbox):
            if not next(counter):
                continue
            img = grab(sct, bbox)
            video_writer.write(img)
        print('Finished recording')

# Capture images from a PSD file after it is written to disk.
async def run_psd_capture(config):
    with ZipFile(config.frame_data, 'a', zipfile.ZIP_DEFLATED) as zfile, Metadata(zfile) as metadata:
        print(f'Frame data: {config.frame_data}')
        async for _ in get_file_tracker_events(config.psd_file):
            while True:
                try:
                    img = PSDImage.open(config.psd_file).composite()
                except:
                    await asyncio.sleep(0.25)
                    continue
                else:
                    break
            img.thumbnail((config.image_size_limit, config.image_size_limit))
            metadata.update_max_size(img.size)
            with zfile.open(str(time.time_ns()), 'w') as mfile:
                img.save(mfile, format='jpeg', quality=95)
        print('Finished recording')

# Attempt to hit a save hotkey in a target app when there is a break in activity.
async def run_auto_save_app(config):
    window, bbox = get_window_and_bbox(config)
    if window is None:
        return
    if bbox is None:
        bbox = window.bbox
    keyboard = pynput.keyboard.Controller()
    save_task = None
    async def save():
        while not window.isActive:
            await asyncio.sleep(0.1)
        keyboard.type(config.save_key)
    async def delay_save():
        nonlocal save_task
        await asyncio.sleep(config.save_delay)
        save_task = asyncio.create_task(save())
    delay_task = None
    async for _ in get_input_tracker_events(window, bbox):
        if save_task is not None:
            await save_task
        if delay_task is not None:
            delay_task.cancel()
        delay_task = asyncio.create_task(delay_save())

async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame-data', help='Name of the file to store frames in (zip file).')
    parser.add_argument('--export', action='store_true', help='Export the given frame data file to an MP4.')
    parser.add_argument('--sai', action='store_true', help='Read SAI\'s memory directly to capture frames. Prompts for canvas selection.')
    parser.add_argument('--psd-file', help='Instead of screen recording, record the specified PSD file every time it is written to.')
    parser.add_argument('--image-size-limit', type=int, default=1000, help='Limit the resolution from a PSD or SAI capture.')
    parser.add_argument('--export-time-limit', type=float, default=60, help='Compress the play time of the exported MP4 file to be no longer than the given seconds. Default is 60 seconds. Set to 0 for uncompressed play time.')
    parser.add_argument('--nth-frame', type=int, default=1, help='For screen recording, record only every Nth frame.')
    parser.add_argument('--drag-grab', action='store_true', help='Drag a rectangle over a window to capture that area. Useful for when a subwindow cannot be captured by click.')
    parser.add_argument('--use-last-grab', action='store_true', help='Use the last drag rectangle stored in the frames file, otherwise you will be prompted to grab a new area.')
    parser.add_argument('--video-file', help='Video file output for recording directly to video (not used by export).')
    parser.add_argument('--convert', action='store_true', help='Convert a video-file to a time compressed shorter video within the given export-time-limit. This is useful when using video-file only output.')
    parser.add_argument('--fps', type=int, default=30, help='FPS of an exported video.')
    parser.add_argument('--auto-save-app', action='store_true', help='Attempt to auto save the target program by sending it the configured save key when it\'s focused.')
    parser.add_argument('--save-delay', type=float, default=1, help='Delay after an action is made before auto-saving target program for PSD recording.')
    parser.add_argument('--save-key', type=str, default='o', help='Hotkey used to save target program.')
    config = parser.parse_args()
    no_frame_data = config.frame_data is None
    if no_frame_data:
        if config.psd_file is not None:
            config.frame_data = Path(config.psd_file).with_suffix('.zip')
            no_frame_data = False
        else:
            config.frame_data = f'{time.time_ns()}.zip'
    config.frame_data = Path(config.frame_data)
    config.frame_data.parent.mkdir(parents=True, exist_ok=True)
    print('Press Ctrl+C here to stop at any time')
    if config.export:
        if no_frame_data:
            print('frame-data or psd-file must be specified for export')
        else:
            run_export(config)
    elif config.convert:
        if config.video_file is None:
            print('video-file must be specified for conversion')
        else:
            config.video_file = Path(config.video_file)
            run_convert_video(config)
    elif config.sai:
        await run_sai_capture_to_frames(config)
    elif config.psd_file is not None:
        config.psd_file = Path(config.psd_file)
        capture_task = asyncio.create_task(run_psd_capture(config))
        if config.auto_save_app:
            await run_auto_save_app(config)
            capture_task.cancel()
        await capture_task
    elif config.video_file is not None:
        config.video_file = Path(config.video_file).with_suffix('.mp4')
        await run_capture_to_video(config)
    else:
        await run_capture_to_frames(config)

def main():
    asyncio.run(async_main())

if __name__ == '__main__':
    main()