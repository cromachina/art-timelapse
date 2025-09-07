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

FPS = 30

def even_dim(a):
    return a if a % 2 == 0 else a - 1

def even_size(size):
    return tuple(even_dim(int(x)) for x in size)

def max_size(size_a, size_b):
    return tuple(max(a, b) for a, b in zip(size_a, size_b))

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

def get_window_and_bbox(config):
    if config.drag_grab:
        return drag_window_and_bbox()
    else:
        return get_window_from_click(), None

def get_root_window(window):
    pid = window.getPID()
    windows = { w.getHandle(): w for w in pywinctl.getAllWindows() if w.getPID() == pid }
    while True:
        phandle = window.getParent()
        if phandle in windows:
            window = windows[phandle]
        else:
            return window

def get_window_from_pid(pid):
    result = None
    for window in pywinctl.getAllWindows():
        if window.getPID() == pid:
            result = window
    if result is None:
        print('Could not find window automatically')
        result = get_window_from_click()
    if result is not None:
        result = get_root_window(result)
    return result

# Metadata handler for zipfiles.
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

# Handles output to a video file.
class VideoWriter:
    def __init__(self, file_path, size, container, codec, log=False):
        self.size = even_size(size)
        self.file_path = file_path.with_suffix(f'.{container}')
        if log:
            print(f'Writing to video: {self.file_path} ({codec})')
        self.writer = cv2.VideoWriter(str(self.file_path), cv2.VideoWriter.fourcc(*codec), FPS, self.size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        self.writer.release()

    def write(self, img, cvtColor=True):
        if img.size != self.size:
            img = ImageOps.pad(img, self.size)
        img = np.asarray(img)
        if cvtColor:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.writer.write(img)

    def write_numpy(self, data):
        self.writer.write(data)

# Handles input from a video file.
class VideoReader:
    def __init__(self, file_path):
        self.video_reader = cv2.VideoCapture(str(file_path))

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        self.video_reader.release()

    def read(self):
        ret, data = self.video_reader.read()
        return data if ret else None

    def get_frame_count(self):
        return int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_size(self):
        return int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Handle writing frames to a zipfile with metadata.
class ZipfileFramesWriter:
    def __init__(self, frames, **_):
        filename = frames.with_suffix('.zip')
        filename.parent.mkdir(parents=True, exist_ok=True)
        print(f'Writing to frames zip: {filename}')
        self.zfile = ZipFile(filename, 'a', zipfile.ZIP_DEFLATED)
        self.metadata = Metadata(self.zfile)

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        self.metadata.save()
        self.zfile.close()
        print(f'Closing frames zip: {self.zfile.filename}')

    def write(self, img):
        self.metadata.update_max_size(img.size)
        with self.zfile.open(f'{time.time_ns()}.jpg', 'w') as mfile:
            img.save(mfile, format='jpeg', quality=95)

# Automatically open and close video writers if the input images change size.
class VideoSequenceWriter:
    def __init__(self, frames, container, codec, **_):
        self.container = container
        self.codec = codec
        self.folder = Path(frames)
        print(f'Writing to frames folder: {self.folder}; container: {container}; codec: {codec}')
        self.folder.mkdir(parents=True, exist_ok=True)
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def open_new(self, size):
        self.writer = VideoWriter(self.folder / f'{time.time_ns()}', size, self.container, self.codec)

    def check_new(self, size):
        if self.writer is None:
            self.open_new(size)
        elif even_size(size) != self.writer.size:
            self.writer.close()
            self.open_new(size)

    def write(self, img):
        self.check_new(img.size)
        self.writer.write(img)

# Read a folder of mp4 files as if it were a single contiguous file.
class VideoSequenceReader:
    def __init__(self, folder):
        self.folder = Path(folder)
        print(f'Reading from frames folder: {self.folder}')
        self.reader = None
        self.frame_count = 0
        self.size = (0, 0)
        self.files = []
        self.file_index = 0
        for file in sorted(self.folder.glob('*')):
            reader = VideoReader(file)
            sub_frame_count = reader.get_frame_count()
            if sub_frame_count > 0:
                self.files.append(file)
                self.frame_count += sub_frame_count
                self.size = max_size(self.size, reader.get_size())
        self.load_next_video()

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        if self.reader is not None:
            self.reader.close()

    def get_last_frame(self):
        reader = VideoReader(self.files[-1])
        last_frame = None
        while True:
            data = reader.read()
            if data is not None:
                last_frame = data
            else:
                break
        reader.close()
        return last_frame

    def load_next_video(self):
        if self.file_index < len(self.files):
            if self.reader is not None:
                self.reader.close()
            self.reader = VideoReader(self.files[self.file_index])
            self.file_index += 1

    def read(self):
        data = self.reader.read()
        if data is None:
            self.load_next_video()
            data = self.reader.read()
        return data

class EventTrackerBase:
    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.stop()

    def emit_event(self):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def stop_event_stream(self):
        self._queue.shutdown()

    async def get_event(self):
        await self._queue.get()

    async def get_event_stream(self):
        try:
            while True:
                yield await self.get_event()
        except asyncio.CancelledError:
            return

# Track mouse clicks on a window.
class InputTracker(EventTrackerBase):
    def __init__(self, target_window, bbox=None):
        super().__init__()
        self._title = target_window.title
        self.target_window = target_window
        self.bbox = bbox
        self.click_started_in_window = False
        self.start()

    def on_click_callback(self, x, y, button, is_pressed):
        if is_pressed:
            window = pywinctl.getTopWindowAt(x, y)
            self.click_started_in_window = window.getHandle() == self.target_window.getHandle()
            if self.click_started_in_window:
                bbox = window.rect if self.bbox is None else self.bbox
                self.click_started_in_window = point_in_bbox(bbox, (x, y))
        if not is_pressed and self.click_started_in_window:
            self.click_started_in_window = False
            self.emit_event()

    async def track_window(self):
        while self.target_window.isAlive:
            await asyncio.sleep(0.1)
        print(f'Window lost: {self._title}')
        self.stop_event_stream()

    def start(self):
        print(f'Tracking input for window: {self.target_window.title}')
        self.mouse_listener = pynput.mouse.Listener(on_click=self.on_click_callback)
        self.mouse_listener.start()
        self._wait_task = asyncio.create_task(self.track_window())

    def stop(self):
        self.mouse_listener.stop()
        print(f'Stopped tracking window: {self.target_window.title}')

# Track a file and when it gets saved/closed.
class FileTracker(FileSystemEventHandler, EventTrackerBase):
    def __init__(self, target_file):
        FileSystemEventHandler.__init__(self)
        EventTrackerBase.__init__(self)
        self.target_file = target_file
        self.observer = Observer()
        self.start()

    def on_closed(self, event):
        if Path(event.src_path) == self.target_file:
            self.emit_event()

    def start(self):
        print(f'Tracking file: {self.target_file}')
        self.observer.schedule(self, self.target_file.parent)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()
        print(f'Stopped tracking file: {self.target_file}')

# Filter a sequence of frame indices achieve a specific maximum video length.
def filter_frames(config, frames):
    if config.export_time_limit > 0:
        frame_count = len(frames)
        target_frames = int(config.export_time_limit * FPS)
        if frame_count > target_frames:
            nth = frame_count / target_frames
            return [frames[round(i * nth)] for i in range(target_frames)]
    return frames

# Pick the appropriate export type based on the target frames path.
def run_export(config):
    path = Path(config.frames)
    zip_path = path.with_suffix('.zip')
    if zip_path.exists() and zip_path.is_file():
        run_export_from_zip(config)
    elif path.exists() and path.is_dir():
        run_export_from_video_dir(config)
    else:
        raise Exception(f'No appropriate frame data found for {config.frames}')

# Export frames captured to a zip file to video.
def run_export_from_zip(config):
    path = Path(config.frames).with_suffix('.zip')
    with ZipFile(path, 'r') as zfile, Metadata(zfile) as metadata:
        frames = sorted(zfile.namelist())
        if len(frames) == 0:
            return
        frames.insert(0, frames[-1])
        frames = filter_frames(config, frames)
        with VideoWriter(config.frames, metadata.get_max_size(), config.container, config.codec, log=True) as writer:
            for frame in tqdm.tqdm(frames, unit='frames'):
                with zfile.open(frame, 'r') as mfile:
                    img = Image.open(mfile)
                    writer.write(img)

# Concatenate videos captured to a folder.
def run_export_from_video_dir(config):
    with VideoSequenceReader(config.frames) as reader:
        if reader.frame_count == 0:
            return
        last_frame = Image.fromarray(reader.get_last_frame())
        frames = list(range(reader.frame_count))
        frames = filter_frames(config, frames)
        index = 0
        with VideoWriter(config.frames, reader.size, config.container, config.codec, log=True) as writer:
            writer.write(last_frame, False)
            for frame in tqdm.tqdm(frames, unit='frames'):
                while index < frame:
                    reader.read()
                    index += 1
                data = reader.read()
                data = Image.fromarray(data)
                index += 1
                writer.write(data, False)

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
        if len(canvas_list) == 0:
            print('No open canvases detected.')
            return None
        print('Select a canvas to record (Ctrl+C to cancel):')
        for i, canvas in zip(range(len(canvas_list)), canvas_list):
            print(f'[{i + 1}] {canvas.get_name()} ({canvas.get_short_path()})')
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
    return (a is None or b is None) or (a.shape != b.shape) or (np.any(a != b))

def get_writer(config):
    if config.zip:
        writer = ZipfileFramesWriter
    else:
        writer = VideoSequenceWriter
    return writer(**vars(config))

# Capture images directly from SAI.
async def run_sai_capture(config):
    with get_writer(config) as writer, sai.SAI() as sai_proc:
        if not sai_version_check(sai_proc):
            return
        canvas = select_canvas(sai_proc)
        if canvas is None:
            return
        window = get_window_from_pid(sai_proc.get_pid())
        if window is None:
            return
        last_img = None
        async for _ in InputTracker(window).get_event_stream():
            if not sai_proc.check_if_canvas_exists(canvas):
                print('Canvas lost, stopping now')
                break
            img = sai_proc.get_canvas_image(canvas)
            if not is_different_image(last_img, img):
                continue
            last_img = img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((config.image_size_limit, config.image_size_limit))
            writer.write(img)

# Capture a window's rectangular subregion or subwindow.
async def run_screen_capture(config):
    with get_writer(config) as writer, mss() as sct:
        window, bbox = get_window_and_bbox(config)
        if window is None:
            return
        if output is VideoSequenceWriter and bbox is None:
            bbox = window.bbox
        async for _ in InputTracker(window, bbox).get_event_stream():
            img = grab(sct, bbox)
            writer.write(img)

# Capture images from a PSD file after it is written to disk.
async def run_psd_capture(config):
    with get_writer(config) as writer:
        async for _ in FileTracker(config.psd_file).get_event_stream():
            while True:
                try:
                    img = PSDImage.open(config.psd_file).composite()
                except:
                    await asyncio.sleep(0.25)
                    continue
                else:
                    break
            img.thumbnail((config.image_size_limit, config.image_size_limit))
            writer.write(img)

async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', help='Name of the file or directory to store frames or videos.')
    parser.add_argument('--sai', action='store_true', help='Read SAI\'s memory directly to capture frames. Prompts for canvas selection.')
    parser.add_argument('--psd-file', help='Instead of screen recording, record the specified PSD file every time it is written to.')
    parser.add_argument('--image-size-limit', type=int, default=1000, help='Limit the resolution from a PSD or SAI capture.')
    parser.add_argument('--export', action='store_true', help='Export the given frame data file to an MP4, or concatenate a video directory. Figures out what to do based on the frames path.')
    parser.add_argument('--export-time-limit', type=float, default=60, help='Compress the play time of the exported MP4 file to be no longer than the given seconds. Default is 60 seconds. Set to 0 for uncompressed play time.')
    parser.add_argument('--drag-grab', action='store_true', help='Drag a rectangle over a window to capture that area. Useful for when a subwindow cannot be captured by click.')
    parser.add_argument('--container', default='webm', help='The format to store video in, for example "webm", "mp4", "avi". Default for storage is webm with VP8 codec for better color quality, but it takes up more space than mp4 with avc1.')
    parser.add_argument('--codec', default='vp80', help='The fourcc for the video container\'s codec. Default for storage is VP8 (vp80).')
    parser.add_argument('--web', action='store_true', help='Sets the video container and codec to "mp4" and "avc1" respectively, which is what Twitter and many other websites expect for video uploads. Use this with --export.')
    parser.add_argument('--zip', action='store_true', help='If outputting to a zip file of JPEGs instead of a folder of videos.')
    config = parser.parse_args()
    if config.web:
        config.container = 'mp4'
        config.codec = 'avc1'
    no_frames = config.frames is None
    if no_frames:
        if config.psd_file is not None:
            config.frames = Path(config.psd_file).with_suffix('')
            no_frames = False
        else:
            config.frames = Path(f'{time.time_ns()}')
    config.frames = Path(config.frames)
    print('Press Ctrl+C here to stop at any time')
    if config.export:
        if no_frames:
            print('--frames or --psd-file must be specified for export')
        else:
            run_export(config)
    elif config.sai:
        await run_sai_capture(config)
    elif config.psd_file is not None:
        config.psd_file = Path(config.psd_file)
        await run_psd_capture(config)
    else:
        await run_screen_capture(config)

def main():
    asyncio.run(async_main())

if __name__ == '__main__':
    main()