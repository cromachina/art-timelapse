import asyncio
import logging
from pathlib import Path
import json
from collections import deque
import threading
import traceback
import time

import ttkbootstrap as ttk
from ttkbootstrap import tooltip
from tkinter import filedialog

from . import asynctk, timelapse, sai

def get_fixed_font(font_name='Consolas', font_size=10):
    if font_name in ttk.font.families():
        font = (font_name, font_size)
    else:
        if font_name not in ttk.font.names():
            font_name = 'TkFixedFont'
        font = ttk.font.nametofont(font_name)
        font.config(size=font_size)
    return font

class RollingAverage:
    def __init__(self, n=20):
        self.n = n
        self.total = 0
        self.values = deque()

    def put(self, value):
        self.total += value
        self.values.append(value)
        if len(self.values) > self.n:
            self.total -= self.values.popleft()

    def get(self):
        return self.total / len(self.values)

    def next(self, value):
        self.put(value)
        return self.get()

class DefaultIntVar(ttk.IntVar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self):
        try:
            return super().get()
        except:
            return 0

class StackStringVar(ttk.StringVar):
    def __init__(self, *args, **kwargs):
        self.stack = []
        super().__init__(*args, **kwargs)

    def push(self, value):
        self.stack.append(self.get())
        self.set(value)

    def pop(self):
        value = self.stack.pop()
        self.set(value)

class ButtonRow(ttk.Frame):
    def __init__(self, master, text=None, textvariable=None):
        super().__init__(master)
        self.pack(fill=ttk.X)
        self.button = ttk.Button(self, text=text, textvariable=textvariable)
        self.button.pack(side=ttk.LEFT, fill=ttk.X, expand=True)

    def set_text(self, text):
        self.button.config(text=text)

    def set_callback(self, callback):
        self.button.config(command=callback)

class ProgressRow(ttk.Frame):
    def __init__(self, master, font=None):
        super().__init__(master)
        self.pack(fill=ttk.X)
        self.label = ttk.Label(self, font=font)
        self.label.pack()
        self.progressbar = ttk.Progressbar(self)
        self.progressbar.pack(fill=ttk.X, expand=True)
        self.loop = asyncio.get_running_loop()

    def iterate(self, iterable, unit='it'):
        self.last_update = float('-inf')
        self.avg_ips = RollingAverage()
        self.avg_eta_sec = RollingAverage()
        total = len(iterable)
        i = 0
        start_time = time.monotonic()
        delta = 0
        for it in iterable:
            delta_start = time.monotonic()
            yield it
            i += 1
            delta = time.monotonic() - delta_start
            self.loop.call_soon_threadsafe(self.set_value, i, total, unit, start_time, delta)
        self.last_update = float('-inf')
        self.loop.call_soon_threadsafe(self.set_value, i, total, unit, start_time, delta)

    def set_value(self, i, total, unit, start_time, delta):
        value = (i / total) * 100
        now = time.monotonic()
        if now - self.last_update > 0.1:
            self.last_update = now
            elapsed_time = time.monotonic() - start_time
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)
            eta_total = delta * (total - i)
            eta_minutes = int(eta_total // 60)
            eta_seconds = int(self.avg_eta_sec.next(eta_total % 60))
            ips = self.avg_ips.next(1 / delta)
            self.label.config(text=f'{i}/{total} [{elapsed_minutes:02}:{elapsed_seconds:02}<{eta_minutes:02}:{eta_seconds:02}, {ips:6.2f}{unit}/s]')
        self.progressbar.config(value=value)

class LabelRow(ttk.Frame):
    def __init__(self, master, text):
        super().__init__(master)
        self.pack(fill=ttk.X)
        self.label = ttk.Label(self, text=text)
        self.label.pack(side=ttk.LEFT)

class StatusLabelRow(LabelRow):
    def __init__(self, master, label_text, status_text='', textvariable=None):
        super().__init__(master, label_text)
        self.status = ttk.Label(self, text=status_text, textvariable=textvariable)
        self.status.pack(side=ttk.LEFT)

class CheckbuttonLabelRow(LabelRow):
    def __init__(self, master, text, variable=None):
        super().__init__(master, text)
        self.checkbutton = ttk.Checkbutton(self, variable=variable)
        self.checkbutton.pack(side=ttk.LEFT)

def check_digit(p):
    return str(p).isdigit() or str(p) == ''

class EntryLabelRow(LabelRow):
    def __init__(self, master, text, numbers=False, button_text=None, textvariable=None):
        super().__init__(master, text)
        self.textvariable = textvariable or ttk.StringVar()
        kwargs = {}
        if numbers:
            kwargs['validate'] = ttk.ALL
            proc = self.register(check_digit)
            kwargs['validatecommand'] = (proc, '%P')
        kwargs['textvariable'] = self.textvariable
        self.entry = ttk.Entry(self, **kwargs)
        self.entry.pack(side=ttk.LEFT, fill=ttk.X, expand=True)
        self.button = None
        if button_text is not None:
            self.button = ttk.Button(self, text=button_text)
            self.button.pack(side=ttk.LEFT)

    def enable(self, state):
        self.entry.config(state=ttk.NORMAL if state else ttk.DISABLED)

def get_nearest_dir(path):
    path = timelapse.expand_path(path).absolute()
    if path.is_dir():
        return path
    for p in path.parents:
        if p.is_dir():
            return p
    return Path('.')

class FilePickerEntry(EntryLabelRow):
    def __init__(self, master, text, title='Select', mode='open', filetypes=(('Any', '*.*'),), button_text='Select', **kwargs):
        super().__init__(master, text, button_text=button_text, **kwargs)
        self.title = title
        self.mode = mode
        self.filetypes = filetypes
        self.button.config(command=self.pick_file)

    def pick_file(self, *_args):
        initdir = get_nearest_dir(self.textvariable.get())
        match self.mode:
            case 'open':
                filename = filedialog.askopenfilename(
                    title=self.title,
                    initialdir=initdir,
                    filetypes=self.filetypes,
                )
            case 'save':
                filename = filedialog.asksaveasfilename(
                    title=self.title,
                    initialdir=initdir,
                    filetypes=self.filetypes,
                )
            case 'dir':
                filename = filedialog.askdirectory(
                    title=self.title,
                    initialdir=initdir,
                )
            case _:
                return
        if filename is not None:
            self.textvariable.set(filename)

class ComboboxLabelRow(LabelRow):
    def __init__(self, master, text, values=None, index_variable=None):
        super().__init__(master, text)
        self.index_var = index_variable
        if self.index_var is None:
            self.index_var = ttk.IntVar(value=0)
        self.combobox = ttk.Combobox(self, state=ttk.READONLY)
        self.combobox.pack(side=ttk.LEFT, fill=ttk.X, expand=True)
        self.reentering = False
        self.combobox.bind('<<ComboboxSelected>>', self.on_selected)
        self.index_var.trace_add('write', self.var_trace)
        if values is not None:
            self.set_values(values)

    def var_trace(self, *_args):
        if self.reentering:
            return
        if isinstance(self.index_var, ttk.StringVar):
            self.combobox.set(self.index_var.get())
        else:
            self.combobox.current(self.index_var.get())

    def on_selected(self, _event):
        self.reentering = True
        if isinstance(self.index_var, ttk.StringVar):
            self.index_var.set(self.combobox.get())
        else:
            self.index_var.set(self.combobox.current())
        self.reentering = False

    def get_index(self):
        return self.combobox.current()

    def set_values(self, values):
        #value = self.combobox.get()
        self.combobox.config(values=values)
        try:
            if isinstance(self.index_var, ttk.StringVar):
                self.combobox.current(values.index(self.index_var.get()))
            else:
                self.combobox.current(self.index_var.get())
        except:
            if len(values) > 0:
                self.combobox.current(0)
            else:
                self.combobox.set('')

class VideoConfigFrame(ttk.Frame):
    def __init__(self, master, shared_vars):
        super().__init__(master)
        self.pack(fill=ttk.X)
        self.video_types = [
            ('mp4/avc1 - Works with most websites, including Twitter', ('mp4', 'avc1')),
            ('webm/vp80 - Better color quality, but larger file size', ('webm', 'vp80')),
            ('Use custom container/codec', None)
        ]
        self.video_type_var = shared_vars['video_type_var']
        self.custom_container_var = shared_vars['custom_container_var']
        self.custom_codec_var = shared_vars['custom_codec_var']
        self.button_text = shared_vars['button_text']
        self.video_type_box = ComboboxLabelRow(self, 'Video type', values=[t[0] for t in self.video_types], index_variable=self.video_type_var)
        tooltip.ToolTip(self.video_type_box, 'Useful container/codec presets.')
        self.custom_container_entry = EntryLabelRow(self, 'Custom container', textvariable=self.custom_container_var)
        tooltip.ToolTip(self.custom_container_entry, 'If you want to use a container that is not listed above, whatever is supported by OpenCV and FFMPEG.')
        self.custom_codec_entry = EntryLabelRow(self, 'Custom codec', textvariable=self.custom_codec_var)
        tooltip.ToolTip(self.custom_codec_entry, 'If you want to use a codec that is not listed above, whatever is supported by OpenCV and FFMPEG.')
        self.button = ButtonRow(self, textvariable=self.button_text)

    def get_format(self):
        video_format = self.video_types[self.video_type_var.get()][1]
        if video_format is None:
            video_format = self.custom_container_var.get() or 'mp4', self.custom_codec_var.get() or 'avc1'
        return video_format

    def set_button_callback(self, callback):
        self.button.set_callback(callback)

class StatusArea(ttk.LabelFrame):
    def __init__(self, master, font=None):
        super().__init__(master, text='Status')
        self.pack(fill=ttk.BOTH, expand=True)
        self.text = ttk.Text(self, state=ttk.DISABLED, font=font, width=1, height=1)
        self.text.pack(fill=ttk.BOTH, expand=True)

    def append(self, text):
        self.text.config(state=ttk.NORMAL)
        self.text.insert(index=ttk.END, chars=f'{text}\n')
        self.text.see(ttk.END)
        self.text.config(state=ttk.DISABLED)

class AsyncWidgetLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        self.widget = widget
        self.loop = asyncio.get_running_loop()

    def emit(self, record):
        self.loop.call_soon_threadsafe(self.widget.append, self.format(record))

def make_frames_entry(master, textvariable=None):
    frame = FilePickerEntry(master, 'Frames path', mode='dir', textvariable=textvariable)
    tooltip.ToolTip(frame, 'Folder where video recording cuts are saved.')
    return frame

def make_image_size_limit_box(master, textvariable=None):
    frame = EntryLabelRow(master, 'Image size limit (px)', numbers=True, textvariable=textvariable)
    tooltip.ToolTip(frame, 'Captured images will be resized to this if larger than this value.')
    return frame

def make_notebook_frame(notebook, text):
    frame = ttk.Frame(notebook)
    frame.pack(fill=ttk.BOTH, expand=True)
    notebook.add(frame, text=text)
    return frame

def get_widgets_by_type(master, wtype):
    for widget in master.winfo_children():
        if isinstance(widget, wtype):
            yield widget
        yield from get_widgets_by_type(widget, wtype)

def auto_size_label_rows(master):
    label_rows = list(get_widgets_by_type(master, LabelRow))
    max_width = 0
    for row in label_rows:
        max_width = max(max_width, len(row.label['text']) - 4)
    for row in label_rows:
        row.label.config(width=max_width, anchor='e')

class Settings:
    var_types = {
        int: DefaultIntVar,
        str: ttk.StringVar,
        bool: ttk.BooleanVar,
        float: ttk.DoubleVar,
    }

    def __init__(self, file_name):
        self.file_path = Path.home() / file_name
        self.data = {}
        self.vars = {}
        if self.file_path.exists():
            with self.file_path.open() as fp:
                self.data = json.load(fp)
                for k, v in self.data.items():
                    self.get_var(k, v)

    def get_var(self, key, default):
        if key in self.vars:
            return self.vars.get(key)
        var = Settings.var_types.get(type(default), ttk.Variable)(value=default)
        def save(*_args):
            self.data[key] = var.get()
        var.trace_add('write', save)
        self.vars[key] = var
        return var

    def save(self):
        with self.file_path.open('w') as fp:
            json.dump(self.data, fp)

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.save()

class App(asynctk.AsyncTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('art-timelapse')
        self.settings = Settings('.art-timelapse')
        fixed_font = get_fixed_font()

        #########################################################
        # Tk variables
        self.win_size_var = self.settings.get_var('win_size', (600, 600))
        self.frames_file_var = self.settings.get_var('frames_file', '')
        self.recording_button_text_var = StackStringVar(value='Start Recording')
        self.export_button_text_var = StackStringVar(value='Export')
        recording_vars = {
            'video_type_var': self.settings.get_var('recording_type', 0),
            'custom_container_var': self.settings.get_var('recording_custom_container', ''),
            'custom_codec_var': self.settings.get_var('recording_custom_codec', ''),
            'button_text': self.recording_button_text_var,
        }
        export_vars = {
            'video_type_var': self.settings.get_var('export_type', 0),
            'custom_container_var': self.settings.get_var('export_custom_container', ''),
            'custom_codec_var': self.settings.get_var('export_custom_codec', ''),
            'button_text': self.export_button_text_var,
        }
        self.sai_version_status_var = ttk.StringVar()
        self.sai_version_override_var = self.settings.get_var('sai_version_override', 0)
        self.sai_canvas_var = self.settings.get_var('sai_canvas', '')
        self.image_size_limit_var = self.settings.get_var('image_size_limit', 1000)
        self.psd_file_var = self.settings.get_var('psd_file_var', '')
        self.screen_click_button_var = StackStringVar(value='Click window grab and start recording')
        self.screen_grab_button_var = StackStringVar(value='Drag area grab and start recording')
        self.export_file_var = self.settings.get_var('export_file_var', '')
        self.export_time_limit_var = self.settings.get_var('export_time_limit', 60)
        self.export_fps_var = self.settings.get_var('export_fps', 30)
        self.export_use_recording_var = self.settings.get_var('export_user_recording', True)
        self.selected_tab_var = self.settings.get_var('selected_tab', 0)
        self.selected_tab_thread_safe = 0
        meta_config_theme_var = self.settings.get_var('theme', 'darkly')

        #########################################################
        # GUI widgets and layout
        notebook = ttk.Notebook(self)
        notebook.pack(side=ttk.TOP, fill=ttk.X)

        sai_frame = make_notebook_frame(notebook, 'SAI Recording')
        make_frames_entry(sai_frame, self.frames_file_var)
        StatusLabelRow(sai_frame, 'SAI version detected:', textvariable=self.sai_version_status_var)
        self.sai_version_override_box = ComboboxLabelRow(sai_frame, 'SAI version override', values=[api.version_name for api in sai.get_sai_api_list()], index_variable=self.sai_version_override_var)
        self.sai_canvas_box = ComboboxLabelRow(sai_frame, 'Canvas', index_variable=self.sai_canvas_var)
        make_image_size_limit_box(sai_frame, textvariable=self.image_size_limit_var)
        self.sai_recording_frame = VideoConfigFrame(sai_frame, recording_vars)

        psd_frame = make_notebook_frame(notebook, 'PSD Recording')
        make_frames_entry(psd_frame, self.frames_file_var)
        psd_file_entry = FilePickerEntry(psd_frame, 'PSD file', mode='open', filetypes=[('PSD', '.psd .psb'), ('Any', '*.*')], textvariable=self.psd_file_var)
        make_image_size_limit_box(psd_frame, textvariable=self.image_size_limit_var)
        self.psd_recording_frame = VideoConfigFrame(psd_frame, recording_vars)

        screen_frame = make_notebook_frame(notebook, 'Screen Recording')
        make_frames_entry(screen_frame, self.frames_file_var)
        self.screen_recording_frame = VideoConfigFrame(screen_frame, recording_vars)
        self.screen_recording_frame.button.forget()
        self.screen_click_button = ButtonRow(screen_frame, textvariable=self.screen_click_button_var)
        self.screen_grab_button = ButtonRow(screen_frame, textvariable=self.screen_grab_button_var)

        export_frame = make_notebook_frame(notebook, 'Export Video')
        make_frames_entry(export_frame, self.frames_file_var)
        export_file_entry = FilePickerEntry(export_frame, 'Export file', mode='save', textvariable=self.export_file_var)
        export_time_limit_entry = EntryLabelRow(export_frame, 'Export time limit', numbers=True, textvariable=self.export_time_limit_var)
        export_fps_entry = EntryLabelRow(export_frame, 'Export FPS', numbers=True, textvariable=self.export_fps_var)
        export_user_reocrding = CheckbuttonLabelRow(export_frame, 'Use recording video type', variable=self.export_use_recording_var)
        self.export_config_frame = VideoConfigFrame(export_frame, export_vars)
        self.export_progress = ProgressRow(export_frame, font=fixed_font)

        status_area = StatusArea(self, font=fixed_font)
        logger = AsyncWidgetLogger(status_area)
        logging.getLogger().addHandler(logger)

        meta_config = ttk.Frame(self)
        meta_config.pack(fill=ttk.X)
        ttk.Label(meta_config, text='Theme:').pack(side=ttk.LEFT)
        meta_config_theme_box = ttk.Combobox(meta_config, state=ttk.READONLY)
        meta_config_theme_box.pack(side=ttk.LEFT)
        meta_config_theme_box.config(values=ttk.Style().theme_names(), textvariable=meta_config_theme_var)

        tooltip.ToolTip(self.sai_version_override_box, 'If the running SAI version cannot be detected, the selected override will be used.')
        tooltip.ToolTip(self.sai_canvas_box, 'Select which open SAI canvas to record from.')
        tooltip.ToolTip(psd_file_entry, 'PSD/PSB file to record from as it is saved to disk.')
        tooltip.ToolTip(self.screen_click_button, 'Captures the subwindow that was clicked on. Automatically adjusts capture size to the subwindow.')
        tooltip.ToolTip(self.screen_grab_button, 'Captures a fixed area of the subwindow. Drag a rectangle like a screenshot tool.')
        tooltip.ToolTip(export_file_entry, 'Path to the video file to export to. Extension is determined by video type. Leave blank to automatically use the frames path.')
        tooltip.ToolTip(export_time_limit_entry, 'Maximum time the exported video should be. Enter 0 or leave blank for no time limit.')
        tooltip.ToolTip(export_fps_entry, 'Set the FPS of the exported video. 30 is a common default. Lower FPS is better for PSD recording exports (like 5 FPS).')
        tooltip.ToolTip(export_user_reocrding, 'Use the same video options as the recording tabs and ignore the below settings.')

        #########################################################
        # Init and callbacks
        self.geometry('{}x{}'.format(*self.win_size_var.get()))

        notebook.select(notebook.tabs()[self.selected_tab_var.get()])
        def on_notebook_tab_changed(*_args):
            self.selected_tab_thread_safe = notebook.index(notebook.select())
            self.selected_tab_var.set(self.selected_tab_thread_safe)
        notebook.bind('<<NotebookTabChanged>>', on_notebook_tab_changed)

        meta_config_theme_var.trace_add('write', lambda *_: ttk.Style(meta_config_theme_var.get()))
        meta_config_theme_var.set(meta_config_theme_var.get())

        auto_size_label_rows(self)

        logging.info('Supported SAI versions:')
        for api in sorted(sai.sai_api_lookup.values(), key=lambda x: x.version_name):
            logging.info(f'  {api.version_name}')

        self.sai_proc = None
        self.last_pid = None
        self.current_pid = None
        def on_override_changed(*_args):
            self.clear_sai_proc()
            self.set_sai_proc()
        self.sai_version_override_var.trace_add('write', on_override_changed)
        self.sai_recording_frame.set_button_callback(self.recording_wrapper(True, self.record_sai))
        self.psd_recording_frame.set_button_callback(self.recording_wrapper(True, self.record_psd))
        self.screen_click_button.set_callback(self.recording_wrapper(True, self.record_screen, grab=False))
        self.screen_grab_button.set_callback(self.recording_wrapper(True, self.record_screen, grab=True))
        self.export_config_frame.set_button_callback(self.recording_wrapper(False, self.export))

        #########################################################
        # Background tasks
        self.pid_scan_event = threading.Event()
        # PID scan is slow, so run it on another thread.
        self.pid_scan_task = asyncio.create_task(asyncio.to_thread(self.pid_scan_thread))
        self.background_task = asyncio.create_task(self.run_background_task())
        self.operation_thread_running = False
        self.operation_task: asyncio.Task | None = None

    def cleanup(self):
        self.win_size_var.set((self.winfo_width(), self.winfo_height()))
        self.settings.save()
        self.pid_scan_event.set()
        self.operation_thread_running = False
        for task in [self.background_task, self.operation_task]:
            if task is not None:
                task.cancel()
                try:
                    ex = task.exception()
                    logging.exception(ex)
                except:
                    pass

    def clear_sai_proc(self):
        if self.sai_proc is not None:
            self.sai_proc.close()
            self.sai_proc = None
            self.last_pid = None

    def refresh_sai_canvses(self):
        canvases = [f'{canvas.get_name()} ({canvas.get_short_path()})' for canvas in self.sai_proc.get_canvas_list()]
        self.sai_canvas_box.set_values(canvases)

    def pid_scan_thread(self):
        while not self.pid_scan_event.wait(0.25):
            if self.sai_proc is None and self.operation_task is None and self.selected_tab_thread_safe == 0:
                self.current_pid = sai.find_running_sai_pid()

    def set_sai_proc(self):
        if self.sai_proc is not None and self.sai_proc.is_running():
            self.refresh_sai_canvses()
        else:
            self.clear_sai_proc()
            sai_pid = self.current_pid
            if sai_pid is None:
                self.sai_version_status_var.set("SAI is not running")
            elif sai_pid != self.last_pid:
                self.last_pid = sai_pid
                api = sai.get_sai_api_from_pid(sai_pid)
                # Found version
                if api is not None:
                    self.sai_proc = sai.SAI()
                    self.sai_version_status_var.set(self.sai_proc.api.version_name)
                # Override version
                else:
                    version_index = self.sai_version_override_var.get()
                    api = sai.get_sai_api_list()[version_index]
                    self.sai_proc = sai.SAI(api)
                    self.sai_version_status_var.set(f'{self.sai_proc.api.version_name} (Override)')
                self.refresh_sai_canvses()

    async def run_background_task(self):
        while True:
            if self.operation_task is None and self.selected_tab_var.get() == 0:
                try:
                    self.set_sai_proc()
                except:
                    pass
            await asyncio.sleep(0.25)

    def run_operation(self, recording_mode, task, *args, **kwargs):
        if self.operation_task is not None:
            self.operation_task.cancel()
            self.operation_task = None
            self.operation_thread_running = False
            return
        async def subtask():
            if recording_mode:
                start_log = 'Recording started'
                stop_log = 'Recording stopped'
                stop_text = 'Stop Recording'
            else:
                start_log = 'Exporting started'
                stop_log = 'Exporting stopped'
                stop_text = 'Stop Exporting'
            logging.info(start_log)
            svars = [self.recording_button_text_var, self.export_button_text_var,
                self.screen_grab_button_var, self.screen_click_button_var]
            for var in svars:
                var.push(stop_text)
            try:
                await task(*args, **kwargs)
            except asyncio.CancelledError:
                pass
            except Exception as ex:
                logging.exception(ex)
            finally:
                logging.info(stop_log)
                for var in svars:
                    var.pop()
                self.operation_task = None
        self.operation_task = asyncio.create_task(subtask())

    def recording_wrapper(self, recording_mode, task, *args, **kwargs):
        return lambda: self.run_operation(recording_mode, task, *args, **kwargs)

    async def record_sai(self):
        if self.sai_proc is None:
            logging.info("No valid SAI process found")
            return
        with sai.SAI(type(self.sai_proc.api)) as sai_proc:
            frames_path = self.frames_file_var.get()
            container, codec = self.sai_recording_frame.get_format()
            image_size_limit = self.image_size_limit_var.get()
            canvases = sai_proc.get_canvas_list()
            if len(canvases) == 0:
                logging.info("There are no canvases open")
                return
            canvas = canvases[self.sai_canvas_box.get_index()]
            await timelapse.sai_capture(sai_proc, canvas, image_size_limit, frames_path, container, codec)

    async def record_psd(self):
        frames_path = self.frames_file_var.get()
        container, codec = self.sai_recording_frame.get_format()
        image_size_limit = self.image_size_limit_var.get()
        psd_file = self.psd_file_var.get()
        await timelapse.psd_capture(psd_file, image_size_limit, frames_path, container, codec)

    async def record_screen(self, grab):
        window, bbox = await timelapse.get_window_and_bbox(grab)
        if window is None:
            return
        frames_path = self.frames_file_var.get()
        container, codec = self.sai_recording_frame.get_format()
        await timelapse.screen_capture(window, bbox, frames_path, container, codec)

    async def export(self):
        frames_path = self.frames_file_var.get()
        export_time_limit = self.export_time_limit_var.get()
        export_fps = self.export_fps_var.get()
        if self.export_use_recording_var.get():
            container, codec = self.sai_recording_frame.get_format()
        else:
            container, codec = self.export_config_frame.get_format()
        output_path = self.export_file_var.get()
        def progress_kill_check(iterable, unit):
            for it in self.export_progress.iterate(iterable, unit):
                yield it
                if not self.operation_thread_running:
                    logging.info('Export cancelled')
                    break
        self.operation_thread_running = True
        # Export can have long CPU-bound sections, so run it on another thread.
        await asyncio.to_thread(timelapse.export, progress_kill_check, export_time_limit, export_fps, frames_path, container, codec, output_path)

    def report_callback_exception(self, exc_type, exc_value, exc_tb):
        logging.exception("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

async def main():
    await App().async_main_loop()