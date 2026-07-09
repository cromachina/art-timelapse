from pathlib import Path
import asyncio
import time
import tkinter as tk
import logging
import sys
import importlib.metadata

from PIL import Image, ImageTk
from psd_tools import PSDImage
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import cv2
import numpy as np
import pynput
import pywinctl
from pywinbox import Point, Size, Rect
from mss import mss

from . import asynctk, sai, _

__version__ = importlib.metadata.version('art-timelapse')

FPS = 30

# MSS has issues being created and destroyed multiple times on Linux.
sct = mss()

def get_size(img:np.ndarray) -> Size:
    return Size(img.shape[1], img.shape[0])

def get_channels(img:np.ndarray) -> int:
    return img.shape[2]

def to_shape(size:Size, channels) -> tuple[int, int, int]:
    return size.height, size.width, channels

def grab_numpy(sct, bbox) -> np.ndarray:
    return np.array(sct.grab(bbox))[:,:,:3]

def grab(sct, bbox) -> Image.Image:
    img = sct.grab(bbox)
    return Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')

def get_fit_size(input_size:Size, output_size:Size) -> Size:
    output_size = output_size
    result_w = output_size.width
    result_h = int(output_size.width / input_size.width * input_size.height)
    if result_h > output_size.height:
        result_h = output_size.height
        result_w = int(output_size.height / input_size.height * input_size.width)
    return Size(result_w, result_h)

def get_reuse_array(storage:dict, key:str, shape:tuple) -> np.ndarray:
    if storage is None:
        return None
    if key in storage:
        array = storage[key]
        if array.shape != shape:
            array = np.empty(shape, dtype=np.uint8)
            storage[key] = array
        return array
    else:
        array = np.empty(shape, dtype=np.uint8)
        storage[key] = array
        return array

def image_fit_and_pad(image:np.ndarray, size:Size, interpolation=cv2.INTER_AREA, reuse_arrays:dict=None) -> np.ndarray:
    image_size = get_size(image)
    if image_size == size:
        return image
    fit_size = get_fit_size(image_size, size)
    fit_top = (size.height - fit_size.height) // 2
    fit_bottom = size.height - fit_size.height - fit_top
    fit_left = (size.width - fit_size.width) // 2
    fit_right = size.width - fit_size.width - fit_left
    if fit_size == image_size:
        resize_dst = image
    else:
        resize_dst = get_reuse_array(reuse_arrays, 'resize', to_shape(fit_size, get_channels(image)))
        resize_dst = cv2.resize(image, fit_size, resize_dst, interpolation=interpolation)
    border_dst = get_reuse_array(reuse_arrays, 'border', to_shape(size, get_channels(image)))
    return cv2.copyMakeBorder(resize_dst, fit_top, fit_bottom, fit_left, fit_right, cv2.BORDER_CONSTANT, border_dst)

def image_thumbnail(image:np.ndarray, size:Size, dst=None, interpolation=cv2.INTER_AREA):
    image_size = get_size(image)
    if image_size.width <= size.width and image_size.height <= size.height:
        return image
    fit_size = get_fit_size(image_size, size)
    if dst is not None and get_size(dst) != fit_size:
        dst = None
    return cv2.resize(image, fit_size, dst=dst, interpolation=interpolation)

def is_image_equal(a:np.ndarray | None, b:np.ndarray | None) -> bool:
    return (a is not None and b is not None) and (a.shape == b.shape) and (np.all(a == b))

def expand_path(path:Path) -> Path:
    path = Path(path)
    try:
        path = path.expanduser()
    except:
        pass
    return path

def even_dim(a:int) -> int:
    return a if a % 2 == 0 else a + 1

def even_size(size:Size) -> Size:
    return Size(even_dim(int(size.width)), even_dim(int(size.height)))

def max_size(size_a:Size, size_b:Size) -> Size:
    return Size(max(size_a.width, size_b.width), max(size_a.height, size_b.height))

def point_in_bbox(bbox:Rect, point:Point):
    return bbox.left <= point.x <= bbox.right and bbox.bottom <= point.y <= bbox.top

def nth_counter(nth:int):
    counter = 1
    while True:
        if counter == 1:
            counter = nth
            yield True
        else:
            counter -= 1
            yield False

def get_spanned_rect(pos1:Point, pos2:Point) -> Rect:
    p1x, p1y = pos1
    p2x, p2y = pos2
    return Rect(min(p1x, p2x), min(p1y, p2y), max(p1x, p2x), max(p1y, p2y))

class WindowGrabber(tk.Toplevel):
    def __init__(self, drag_area, *args, **kwargs):
        if tk._default_root is None:
            self.root = asynctk.AsyncTk()
            self.root.withdraw()
            self.root_task = asyncio.create_task(self.root.async_main_loop())
        else:
            self.root = None
        self.event = asyncio.Event()
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
        self.canvas = tk.Canvas(self, width=self.sw, height=self.sh, highlightthickness=0)
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
            logging.info(_('Click and drag on a subwindow to start tracking an area. Right click to cancel.'))
            self.bind('<Motion>', self.drag_motion)
            self.bind('<ButtonPress-1>', self.drag_clicked)
            self.bind('<ButtonRelease-1>', self.released)
            self.bind('<ButtonPress-3>', self.set_cancelled)
        else:
            logging.info(_('Click on a subwindow to start tracking. Right click to cancel.'))
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
        if self.root is not None:
            self.root.stop()
            self.root.destroy()
        self.event.set()

    def set_cancelled(self, event):
        logging.info(_('Window grab cancelled'))
        self.cancelled = True
        self.released(event)

    def scan_motion(self, event):
        window = pywinctl.getTopWindowAt(event.x, event.y)
        if window is not None:
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

    async def get_window_and_bbox(self):
        await self.event.wait()
        if self.cancelled:
            return None, None
        window = pywinctl.getTopWindowAt(*self.pos1)
        bbox = Rect(*self.pos1, *self.pos2) if self.drag_area else None
        return window, bbox

async def get_window_and_bbox(drag_grab):
    return await WindowGrabber(drag_grab).get_window_and_bbox()

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
        logging.info(_('Could not find window automatically'))
        result, _bbox = await get_window_and_bbox(False)
    if result is not None:
        result = get_root_window(result)
    return result

# Handles output to a video file.
class VideoWriter:
    def __init__(self, file_path:Path, size:Size, container:str, codec:str, fps=FPS, log=False):
        self.size = even_size(size)
        self.file_path = expand_path(file_path).with_suffix(f'.{container}')
        if log:
            logging.info(_('Writing to video:') + f' {self.file_path} ({codec})')
        self.writer = cv2.VideoWriter(str(self.file_path), cv2.VideoWriter.fourcc(*codec), fps, self.size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        self.writer.release()

    def write(self, img:np.ndarray, cvt_color=False, reuse_arrays:dict=None):
        img = image_fit_and_pad(img, self.size, reuse_arrays=reuse_arrays)
        if cvt_color:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.writer.write(img)

# Handles input from a video file.
class VideoReader:
    def __init__(self, file_path:Path):
        self.video_reader = cv2.VideoCapture(str(expand_path(file_path)))
        self.data = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        self.video_reader.release()

    def read(self):
        ret, self.data = self.video_reader.read(self.data)
        return self.data if ret else None

    def get_frame_count(self):
        return int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_size(self) -> Size:
        return Size(int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Automatically open and close video writers if the input images change size.
class VideoSequenceWriter:
    def __init__(self, frames_path:Path, container:str, codec:str, frames_auto_split_count=500):
        self.container = container
        self.codec = codec
        self.folder = expand_path(frames_path)
        self.frames_auto_split_count = frames_auto_split_count
        self.frames_count = 0
        logging.info(_('Writing to frames folder:') + f' {self.folder}; container: {container}; codec: {codec}')
        self.folder.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.reuse_arrays = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def open_new(self, size:Size):
        self.writer = VideoWriter(self.folder / f'{time.time_ns()}', size, self.container, self.codec)
        self.frames_count = 0

    def check_new(self, size:Size):
        if self.writer is None:
            self.open_new(size)
        elif even_size(size) != self.writer.size or (self.frames_auto_split_count != 0 and self.frames_count >= self.frames_auto_split_count):
            self.writer.close()
            self.open_new(size)

    def write(self, img:np.ndarray, cvt_color=False):
        self.check_new(get_size(img))
        self.writer.write(img, cvt_color, reuse_arrays=self.reuse_arrays)
        self.frames_count += 1

# Read a folder of mp4 files as if it were a single contiguous file.
class VideoSequenceReader:
    def __init__(self, folder:Path):
        self.folder = expand_path(folder)
        logging.info(_('Reading from frames folder:') + f' {self.folder}')
        self.reader = None
        self.frame_count = 0
        self.size = Size(0, 0)
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
        try:
            if is_pressed:
                window = pywinctl.getTopWindowAt(x, y)
                if window is None:
                    return
                self.click_started_in_window = window.getHandle() == self.target_window.getHandle()
                if self.click_started_in_window:
                    bbox = window.rect if self.bbox is None else self.bbox
                    self.click_started_in_window = point_in_bbox(bbox, (x, y))
            if not is_pressed and self.click_started_in_window:
                self.click_started_in_window = False
                self.emit_event()
        except:
            pass

    async def track_window(self):
        while self.target_window.isAlive:
            try:
                self._title = self.target_window.title
            except:
                pass
            await asyncio.sleep(0.1)
        logging.info(_('Window lost:') + f' {self._title}')
        self.stop_event_stream()

    def start(self):
        logging.info(_('Tracking input for window:') + f' {self._title}')
        self.mouse_listener = pynput.mouse.Listener(on_click=self.on_click_callback)
        self.mouse_listener.start()
        self._wait_task = asyncio.create_task(self.track_window())

    def stop(self):
        self.mouse_listener.stop()
        self._wait_task.cancel()
        logging.info(_('Stopped tracking window:') + f' {self._title}')

# Track a file and when it gets saved/closed.
class FileTracker(FileSystemEventHandler, EventTrackerBase):
    def __init__(self, target_file:Path):
        FileSystemEventHandler.__init__(self)
        EventTrackerBase.__init__(self)
        self.target_file = expand_path(target_file)
        self.observer = Observer()
        self.start()

    def on_closed(self, event):
        if Path(event.src_path) == self.target_file:
            self.emit_event()

    def start(self):
        logging.info(_('Tracking file:') + f' {self.target_file}')
        self.observer.schedule(self, self.target_file.parent)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()
        logging.info(_('Stopped tracking file:') + f' {self.target_file}')

# Get frame indices achieve a specific maximum video length.
def get_frame_indices(export_time_limit:float, frame_count:int, fps:int):
    if export_time_limit > 0:
        target_frames = int(export_time_limit * fps)
        if frame_count > target_frames:
            nth = frame_count / target_frames
            return target_frames, (round(i * nth) for i in range(target_frames))
    return frame_count, range(frame_count)

def write_preview_frames(progress_iter:function, reader:VideoSequenceReader, writer:VideoSequenceWriter, preview_duration:float, fps:int, reuse_arrays:dict):
    last_frame = reader.get_last_frame()
    if preview_duration == 0:
        writer.write(last_frame, reuse_arrays=reuse_arrays)
    else:
        frame_count = int(preview_duration * fps)
        for _ in progress_iter(range(frame_count), frame_count, unit='frames'):
            writer.write(last_frame, reuse_arrays=reuse_arrays)

def export(progress_iter:function, export_time_limit:float, preview_last_frame:bool, preview_duration:float, fps:int, frames:Path, container:str, codec:str, output_path=''):
    frames = expand_path(frames)
    if output_path == '':
        output_path = frames
    if not (frames.exists() and frames.is_dir()):
        raise Exception(_('No appropriate frame data found:') + f' {frames}')
    with VideoSequenceReader(frames) as reader:
        if reader.frame_count == 0:
            return
        total_frames, frame_indices = get_frame_indices(export_time_limit, reader.frame_count, fps)
        index = 0
        reuse_arrays = {}
        with VideoWriter(output_path, reader.size, container, codec, fps=fps, log=True) as writer:
            if preview_last_frame:
                write_preview_frames(progress_iter, reader, writer, preview_duration, fps, reuse_arrays)
            for frame in progress_iter(frame_indices, total_frames, unit='frames'):
                while index < frame:
                    reader.read()
                    index += 1
                data = reader.read()
                index += 1
                writer.write(data, reuse_arrays=reuse_arrays)

def copy_into(dst:np.ndarray, src:np.ndarray) -> np.ndarray:
    if dst is not None and dst.shape == src.shape:
        np.copyto(dst, src)
        return dst
    else:
        return src.copy()

async def sai_capture(sai_proc:sai.SAI, canvas, image_size_limit:int, frames:Path, container:str, codec:str, auto_split_count:int):
    window = await get_window_from_pid(sai_proc.get_pid())
    image_size_limit_size = Size(image_size_limit, image_size_limit)
    thumbnail = None
    if window is None:
        return
    with VideoSequenceWriter(frames, container, codec, auto_split_count) as writer, InputTracker(window) as tracker:
        last_img = None
        async for _event in tracker.get_event_stream():
            if not sai_proc.api.check_if_canvas_exists(canvas):
                logging.info(_('Canvas lost, stopping now'))
                break
            map_level = sai_proc.api.get_map_level_for_size(canvas, image_size_limit)
            img = sai_proc.api.get_canvas_image(canvas, map_level)
            if is_image_equal(last_img, img):
                continue
            last_img = copy_into(last_img, img)
            thumbnail = image_thumbnail(img, image_size_limit_size, thumbnail)
            writer.write(thumbnail)

async def screen_capture(window:pywinctl.Window, bbox:Rect, frames:Path, container:str, codec:str, auto_split_count:int):
    with VideoSequenceWriter(frames, container, codec, auto_split_count) as writer, InputTracker(window, bbox) as tracker:
        async for _event in tracker.get_event_stream():
            img = grab_numpy(sct, window.rect if bbox is None else bbox)
            writer.write(img)

async def psd_capture(psd_file:Path, image_size_limit:int, frames:Path, container:str, codec:str, auto_split_count:int):
    psd_file = expand_path(psd_file)
    image_size_limit_size = Size(image_size_limit, image_size_limit)
    with VideoSequenceWriter(frames, container, codec, auto_split_count) as writer, FileTracker(psd_file) as tracker:
        async for _event in tracker.get_event_stream():
            while True:
                try:
                    img = np.array(PSDImage.open(psd_file).composite())
                except:
                    await asyncio.sleep(0.25)
                    continue
                else:
                    break
            img = image_thumbnail(img, image_size_limit_size)
            writer.write(img, cvt_color=True)