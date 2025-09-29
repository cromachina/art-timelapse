from pathlib import Path
import asyncio
import time
import tkinter as tk
import logging
import sys

from PIL import Image, ImageOps, ImageTk
from psd_tools import PSDImage
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import cv2
import numpy as np
import pynput
import pywinctl
import tqdm
from mss import mss

from . import asynctk, sai

FPS = 30

Vec2 = tuple[int, int]

# MSS has issues being created and destroyed multiple times on Linux.
sct = mss()

def grab(sct, bbox):
    img = sct.grab(bbox)
    return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

def expand_path(path):
    path = Path(path)
    try:
        path = path.expanduser()
    except:
        pass
    return path

def even_dim(a:int) -> int:
    return a if a % 2 == 0 else a - 1

def even_size(size:Vec2) -> Vec2:
    return even_dim(int(size[0])), even_dim(int(size[1]))

def max_size(size_a:Vec2, size_b:Vec2) -> Vec2:
    return max(size_a[0], size_b[0]), max(size_a[1], size_b[1])

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

def get_spanned_rect(pos1, pos2):
    p1x, p1y = pos1
    p2x, p2y = pos2
    return min(p1x, p2x), min(p1y, p2y), max(p1x, p2x), max(p1y, p2y)

class WindowGrabber(asynctk.AsyncTk):
    def __init__(self, drag_area, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drag_area = drag_area
        self.overrideredirect(True)
        mon = sct.monitors[0]
        self.screen = grab(sct, mon)
        self.sw = mon['width']
        self.sh = mon['height']
        self.geometry(f'{self.sw}x{self.sh}+0+0')
        self.attributes('-topmost', True)
        self.screen = ImageTk.PhotoImage(self.screen, master=self)
        self.canvas = tk.Canvas(self, width=self.sw, height=self.sh)
        self.canvas.place(x=0, y=0)
        self.canvas.create_image((0, 0), anchor=tk.NW, image=self.screen)
        self.overlay = self.make_rect(self.sw, self.sh, 'gray12')
        self.top_rect = self.make_rect()
        self.left_rect = self.make_rect()
        self.right_rect = self.make_rect()
        self.bottom_rect = self.make_rect()
        self.outline = self.canvas.create_rectangle(0, 0, 0, 0, outline='#7777ff', fill='')
        self.pos1 = None
        self.pos2 = None
        self.window = None
        self.cancelled = False
        self.lift()

        if drag_area:
            logging.info('Click on a subwindow to start tracking. Right click to cancel.')
            self.bind('<Motion>', self.drag_motion)
            self.bind('<ButtonPress-1>', self.drag_clicked)
            self.bind('<ButtonRelease-1>', self.released)
            self.bind('<ButtonPress-3>', self.set_cancelled)
        else:
            logging.info('Click on a subwindow to start tracking. Right click to cancel.')
            self.canvas.delete(self.overlay)
            if 'win' in sys.platform:
                self.bind('<Motion>', self.scan_motion_win)
            else:
                self.bind('<Motion>', self.scan_motion)
            self.bind('<ButtonPress-1>', self.scan_clicked)
            self.bind('<ButtonPress-3>', self.set_cancelled)
            e = tk.Event()
            e.x = self.winfo_pointerx()
            e.y = self.winfo_pointery()
            self.scan_motion(e)

    def make_rect(self, w=0, h=0, stipple='gray50'):
        return self.canvas.create_rectangle(0, 0, w, h, outline='', fill='black', stipple=stipple)

    def set_rect(self, lx, ly, rx, ry):
        self.canvas.coords(self.outline, lx, ly, rx, ry)
        self.canvas.coords(self.top_rect, 0, 0, self.sw, ly)
        self.canvas.coords(self.left_rect, 0, ly, lx, ry)
        self.canvas.coords(self.right_rect, rx, ly, self.sw, ry)
        self.canvas.coords(self.bottom_rect, 0, ry, self.sw, self.sh)

    def drag_motion(self, event):
        if self.pos1 is not None:
            self.pos2 = (event.x, event.y)
            self.set_rect(*get_spanned_rect(self.pos1, self.pos2))

    def drag_clicked(self, event):
        self.canvas.delete(self.overlay)
        self.pos1 = (event.x, event.y)
        self.drag_motion(event)

    def released(self, _event):
        self.destroy()
        self.stop()

    def set_cancelled(self, event):
        logging.info("Window grab cancelled")
        self.released(event)
        self.cancelled = True

    def scan_motion(self, event):
        window = pywinctl.getTopWindowAt(event.x, event.y)
        self.set_rect(*window.rect)

    def scan_motion_win(self, event):
        window = pywinctl.getActiveWindow()
        handle = window.getPID()
        windows = pywinctl.getWindowsAt(event.x, event.y)
        for sub in windows:
            if sub.getPID() != handle:
                window = sub
                break
        self.set_rect(*window.rect)

    def scan_clicked(self, event):
        self.pos1 = event.x, event.y
        self.released(event)

    def get_window_and_bbox(self):
        if self.cancelled:
            return None, None
        window = pywinctl.getTopWindowAt(*self.pos1)
        bbox = (*self.pos1, *self.pos2) if self.drag_area else None
        return window, bbox

async def get_window_and_bbox(drag_grab):
    grabber = WindowGrabber(drag_grab)
    await grabber.async_main_loop()
    return grabber.get_window_and_bbox()

def get_root_window(window):
    pid = window.getPID()
    windows = { w.getHandle(): w for w in pywinctl.getAllWindows() if w.getPID() == pid }
    while True:
        phandle = window.getParent()
        if phandle in windows:
            window = windows[phandle]
        else:
            return window

async def get_window_from_pid(pid):
    result = None
    for window in pywinctl.getAllWindows():
        if window.getPID() == pid:
            result = window
    if result is None:
        logging.info('Could not find window automatically')
        result, _ = await get_window_and_bbox(False)
    if result is not None:
        result = get_root_window(result)
    return result

# Handles output to a video file.
class VideoWriter:
    def __init__(self, file_path, size, container, codec, fps=FPS, log=False):
        self.size = even_size(size)
        self.file_path = expand_path(file_path).with_suffix(f'.{container}')
        if log:
            logging.info(f'Writing to video: {self.file_path} ({codec})')
        self.writer = cv2.VideoWriter(str(self.file_path), cv2.VideoWriter.fourcc(*codec), fps, self.size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        self.writer.release()

    def write(self, img, cvt_color=True):
        if img.size != self.size:
            img = ImageOps.pad(img, self.size)
        img = np.asarray(img)
        if cvt_color:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.writer.write(img)

    def write_numpy(self, data):
        self.writer.write(data)

# Handles input from a video file.
class VideoReader:
    def __init__(self, file_path):
        self.video_reader = cv2.VideoCapture(str(expand_path(file_path)))

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

# Automatically open and close video writers if the input images change size.
class VideoSequenceWriter:
    def __init__(self, frames, container, codec):
        self.container = container
        self.codec = codec
        self.folder = expand_path(frames)
        logging.info(f'Writing to frames folder: {self.folder}; container: {container}; codec: {codec}')
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
        self.folder = expand_path(folder)
        logging.info(f'Reading from frames folder: {self.folder}')
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

    def stop(self):
        pass

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
        except (asyncio.CancelledError, asyncio.QueueShutDown):
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

    def on_click_callback(self, x, y, _button, is_pressed):
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
        logging.info(f'Window lost: {self._title}')
        self.stop_event_stream()

    def start(self):
        logging.info(f'Tracking input for window: {self.target_window.title}')
        self.mouse_listener = pynput.mouse.Listener(on_click=self.on_click_callback)
        self.mouse_listener.start()
        self._wait_task = asyncio.create_task(self.track_window())

    def stop(self):
        self.mouse_listener.stop()
        self._wait_task.cancel()
        logging.info(f'Stopped tracking window: {self.target_window.title}')

# Track a file and when it gets saved/closed.
class FileTracker(FileSystemEventHandler, EventTrackerBase):
    def __init__(self, target_file):
        FileSystemEventHandler.__init__(self)
        EventTrackerBase.__init__(self)
        self.target_file = expand_path(target_file)
        self.observer = Observer()
        self.start()

    def on_closed(self, event):
        if Path(event.src_path) == self.target_file:
            self.emit_event()

    def start(self):
        logging.info(f'Tracking file: {self.target_file}')
        self.observer.schedule(self, self.target_file.parent)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()
        logging.info(f'Stopped tracking file: {self.target_file}')

# Filter a sequence of frame indices achieve a specific maximum video length.
def filter_frames(export_time_limit, frames, fps):
    if export_time_limit > 0:
        frame_count = len(frames)
        target_frames = int(export_time_limit * fps)
        if frame_count > target_frames:
            nth = frame_count / target_frames
            return [frames[round(i * nth)] for i in range(target_frames)]
    return frames

def export(progress_iter, export_time_limit, fps, frames, container, codec, output_path='', **_):
    frames = expand_path(frames)
    if output_path == '':
        output_path = frames
    if not (frames.exists() and frames.is_dir()):
        raise Exception(f'No appropriate frame data found for {frames}')
    with VideoSequenceReader(frames) as reader:
        if reader.frame_count == 0:
            return
        last_frame = Image.fromarray(reader.get_last_frame())
        data_frames = list(range(reader.frame_count))
        data_frames = filter_frames(export_time_limit, data_frames, fps)
        index = 0
        with VideoWriter(output_path, reader.size, container, codec, fps=fps, log=True) as writer:
            writer.write(last_frame, False)
            for frame in progress_iter(data_frames, unit='frames'):
                while index < frame:
                    reader.read()
                    index += 1
                data = reader.read()
                data = Image.fromarray(data)
                index += 1
                writer.write(data, False)

# Concatenate videos captured to a folder.
def cli_export(config):
    export(tqdm.tqdm, **vars(config))

def sai_version_check(sai_proc):
    if not sai_proc.is_sai_version_compatible():
        logging.info('Warning: Capture may not work correctly')
        res = input('Continue? [y/N]').lower()
        return res.startswith('y')
    return True

def select_canvas(sai_proc):
    while True:
        canvas_list = sai_proc.get_canvas_list()
        if len(canvas_list) == 0:
            logging.info('No open canvases detected.')
            return None
        logging.info('Select a canvas to record (Ctrl+C to cancel):')
        for i, canvas in zip(range(len(canvas_list)), canvas_list):
            logging.info(f'[{i + 1}] {canvas.get_name()} ({canvas.get_short_path()})')
        res = input(f'Enter index [1-{len(canvas_list)}]:')
        try:
            res = int(res)
            canvas = canvas_list[res - 1]
        except ValueError:
            logging.info('Could not parse input, trying again')
        except IndexError:
            logging.info('Index out of range, trying again')
        else:
            return canvas

def is_different_image(a, b):
    return (a is None or b is None) or (a.shape != b.shape) or (np.any(a != b))

async def sai_capture(sai_proc, canvas, image_size_limit, frames, container, codec, **_):
    window = await get_window_from_pid(sai_proc.get_pid())
    if window is None:
        return
    with VideoSequenceWriter(frames, container, codec) as writer, InputTracker(window) as tracker:
        last_img = None
        async for _ in tracker.get_event_stream():
            if not sai_proc.check_if_canvas_exists(canvas):
                logging.info('Canvas lost, stopping now')
                break
            img = sai_proc.get_canvas_image(canvas)
            if not is_different_image(last_img, img):
                continue
            last_img = img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((image_size_limit, image_size_limit))
            writer.write(img)

# Capture images directly from SAI.
async def cli_sai_capture(config):
    with sai.SAI() as sai_proc:
        if not sai_version_check(sai_proc):
            return
        canvas = select_canvas(sai_proc)
        if canvas is None:
            return
        await sai_capture(sai_proc, canvas, **vars(config))

async def screen_capture(window, bbox, frames, container, codec, **_):
    with VideoSequenceWriter(frames, container, codec) as writer, InputTracker(window, bbox) as tracker:
        async for _ in tracker.get_event_stream():
            img = grab(sct, window.rect if bbox is None else bbox)
            writer.write(img)

# Capture a window's rectangular subregion or subwindow.
async def cli_screen_capture(config):
    window, bbox = await get_window_and_bbox(config)
    if window is None:
        return
    await screen_capture(window, bbox, **vars(config))

async def psd_capture(psd_file, image_size_limit, frames, container, codec, **_):
    with VideoSequenceWriter(frames, container, codec) as writer, FileTracker(psd_file) as tracker:
        async for _ in tracker.get_event_stream():
            while True:
                try:
                    img = PSDImage.open(psd_file).composite()
                except:
                    await asyncio.sleep(0.25)
                    continue
                else:
                    break
            img.thumbnail((image_size_limit, image_size_limit))
            writer.write(img)

# Capture images from a PSD file after it is written to disk.
async def cli_psd_capture(config):
    await psd_capture(**vars(config))

async def main(config):
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
    logging.info('Press Ctrl+C here to stop at any time')
    if config.export:
        if no_frames:
            logging.info('--frames or --psd-file must be specified for export')
        else:
            cli_export(config)
    elif config.sai:
        await cli_sai_capture(config)
    elif config.psd_file is not None:
        config.psd_file = Path(config.psd_file)
        await cli_psd_capture(config)
    else:
        await cli_screen_capture(config)