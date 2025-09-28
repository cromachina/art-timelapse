import asyncio
import logging
from pathlib import Path
import json
import traceback
import time

import ttkbootstrap as ttk
from tkinter import filedialog

from . import asynctk, timelapse, sai

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
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=ttk.X)
        self.label = ttk.Label(self)
        self.label.pack()
        self.progressbar = ttk.Progressbar(self)
        self.progressbar.pack(fill=ttk.X, expand=True)
        self.loop = asyncio.get_running_loop()

    def iterate(self, iterable, unit='it'):
        self.last_update = float('-inf')
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
            eta_seconds = int(eta_total % 60)
            ips = 1 / delta
            self.label.config(text=f'{i}/{total} [{elapsed_minutes:02}:{elapsed_seconds:02}<{eta_minutes:02}:{eta_seconds:02}, {ips:.2f}{unit}/s]')
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
    path = Path(path).absolute()
    if path.is_dir():
        return path
    for p in Path(path).absolute().parents:
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
        if self.mode == 'open':
            filename = filedialog.askopenfilename(
                title=self.title,
                initialdir=initdir,
                filetypes=self.filetypes,
            )
        elif self.mode == 'save':
            filename = filedialog.asksaveasfilename(
                title=self.title,
                initialdir=initdir,
                filetypes=self.filetypes,
            )
        elif self.mode == 'dir':
            filename = filedialog.askdirectory(
                title='Pick frames directory',
                initialdir=initdir,
            )
        else:
            return
        if filename:
            self.textvariable.set(filename)

class ComboboxLabelRow(LabelRow):
    def __init__(self, master, text, values=(), index_variable=None):
        super().__init__(master, text)
        self.index_var = index_variable
        if self.index_var is None:
            self.index_var = ttk.IntVar(value=0)
        self.combobox = ttk.Combobox(self, state=ttk.READONLY)
        self.combobox.pack(side=ttk.LEFT, fill=ttk.X, expand=True)
        self.combobox.bind('<<ComboboxSelected>>', self.on_selected)
        self.set_values(values)

    def on_selected(self, _event):
        if isinstance(self.index_var, ttk.StringVar):
            self.index_var.set(self.combobox.get())
        else:
            self.index_var.set(self.combobox.current())

    def get_index(self):
        return self.combobox.current()

    def set_values(self, values, init=False):
        value = self.combobox.get()
        self.combobox.config(values=values)
        try:
            if init:
                if isinstance(self.index_var, ttk.StringVar):
                    self.combobox.current(values.index(self.index_var.get()))
                else:
                    self.combobox.current(self.index_var.get())
            else:
                self.combobox.current(values.index(value))
        except (ValueError, IndexError):
            if len(values) > 0:
                self.combobox.current(0)
            else:
                self.combobox.set('')

class VideoConfigFrame(ttk.Frame):
    def __init__(self, master, shared_vars):
        super().__init__(master)
        self.pack(fill=ttk.X)
        self.video_types = [
            ('mp4/avc1', ('mp4', 'avc1')),
            ('webm/vp80', ('webm', 'vp80')),
            ('Use custom', None)
        ]
        self.video_type_var = shared_vars['video_type_var']
        self.custom_container_var = shared_vars['custom_container_var']
        self.custom_codec_var = shared_vars['custom_codec_var']
        self.button_text = shared_vars['button_text']
        self.video_type_box = ComboboxLabelRow(self, 'Video type', values=[t[0] for t in self.video_types], index_variable=self.video_type_var)
        self.custom_container_entry = EntryLabelRow(self, 'Custom container', textvariable=self.custom_container_var)
        self.custom_codec_entry = EntryLabelRow(self, 'Custom codec', textvariable=self.custom_codec_var)
        self.button = ButtonRow(self, textvariable=self.button_text)
        self.video_type_box.combobox.current(0)

    def get_format(self):
        video_format = self.video_types[self.video_type_var.get()][1]
        if video_format is None:
            video_format = self.custom_container_var.get() or 'mp4', self.custom_codec_var.get() or 'avc1'
        return video_format

    def set_button_callback(self, callback):
        self.button.set_callback(callback)

class StatusArea(ttk.LabelFrame):
    def __init__(self, master):
        super().__init__(master, text='Status')
        self.pack(fill=ttk.BOTH, expand=True)
        font_size = 10
        if 'Consolas' in ttk.font.families():
            font = ('Consolas', font_size)
        else:
            font = ttk.font.nametofont('TkFixedFont')
            font.config(size=font_size)
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
    return FilePickerEntry(master, 'Frames path', mode='dir', textvariable=textvariable)

def make_image_size_limit_box(master, textvariable=None):
    return EntryLabelRow(master, 'Image size limit (px)', numbers=True, textvariable=textvariable)

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
        int: ttk.IntVar,
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

        self.win_size_var = self.settings.get_var('win_size', (600, 600))
        self.geometry('{}x{}'.format(*self.win_size_var.get()))

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

        notebook = ttk.Notebook(self)
        notebook.pack(side=ttk.TOP, fill=ttk.X)

        sai_frame = make_notebook_frame(notebook, 'SAI Recording')
        make_frames_entry(sai_frame, self.frames_file_var)
        self.sai_version_status_var = ttk.StringVar()
        StatusLabelRow(sai_frame, 'SAI version detected:', textvariable=self.sai_version_status_var)
        self.sai_canvas_var = self.settings.get_var('sai_canvas', '')
        self.sai_canvas_box = ComboboxLabelRow(sai_frame, 'Canvas', index_variable=self.sai_canvas_var)
        self.image_size_limit_var = self.settings.get_var('image_size_limit', 1000)
        make_image_size_limit_box(sai_frame, textvariable=self.image_size_limit_var)
        self.sai_recording_frame = VideoConfigFrame(sai_frame, recording_vars)

        psd_frame = make_notebook_frame(notebook, 'PSD Recording')
        make_frames_entry(psd_frame, self.frames_file_var)
        self.psd_file_var = self.settings.get_var('psd_file_var', '')
        FilePickerEntry(psd_frame, 'PSD file', mode='open', filetypes=[('PSD', '.psd .psb'), ('Any', '*.*')], textvariable=self.psd_file_var)
        make_image_size_limit_box(psd_frame, textvariable=self.image_size_limit_var)
        self.psd_recording_frame = VideoConfigFrame(psd_frame, recording_vars)

        screen_frame = make_notebook_frame(notebook, 'Screen Recording')
        make_frames_entry(screen_frame, self.frames_file_var)
        self.screen_recording_frame = VideoConfigFrame(screen_frame, recording_vars)
        self.screen_recording_frame.button.forget()
        self.screen_click_button_var = StackStringVar(value='Click window grab and start recording')
        self.screen_click_button = ButtonRow(screen_frame, textvariable=self.screen_click_button_var)
        self.screen_grab_button_var = StackStringVar(value='Drag area grab and start recording')
        self.screen_grab_button = ButtonRow(screen_frame, textvariable=self.screen_grab_button_var)

        export_frame = make_notebook_frame(notebook, 'Export Video')
        make_frames_entry(export_frame, self.frames_file_var)
        self.export_file_var = self.settings.get_var('export_file_var', '')
        FilePickerEntry(export_frame, 'Export file', mode='save', textvariable=self.export_file_var)
        self.export_time_limit_var = self.settings.get_var('export_time_limit', 60)
        EntryLabelRow(export_frame, 'Export time limit', numbers=True, textvariable=self.export_time_limit_var)
        self.export_use_recording_var = self.settings.get_var('export_user_recording', True)
        CheckbuttonLabelRow(export_frame, 'Use recording video type', variable=self.export_use_recording_var)
        self.export_config_frame = VideoConfigFrame(export_frame, export_vars)
        self.export_progress = ProgressRow(export_frame)

        self.selected_tab_var = self.settings.get_var('selected_tab', 0)
        notebook.select(notebook.tabs()[self.selected_tab_var.get()])
        notebook.bind('<<NotebookTabChanged>>', lambda _event: self.selected_tab_var.set(notebook.index(notebook.select())))

        status_area = StatusArea(self)
        logger = AsyncWidgetLogger(status_area)
        logging.getLogger().addHandler(logger)

        meta_config = ttk.Frame(self)
        meta_config.pack(fill=ttk.X)
        ttk.Label(meta_config, text='Theme:').pack(side=ttk.LEFT)
        theme_box = ttk.Combobox(meta_config, state=ttk.READONLY)
        theme_var = self.settings.get_var('theme', 'darkly')
        theme_box.pack(side=ttk.LEFT)
        theme_box.config(values=ttk.Style().theme_names(), textvariable=theme_var)
        theme_box.bind('<<ComboboxSelected>>', lambda _: ttk.Style(theme_box.get()))
        theme_box.set(ttk.Style(theme_var.get()).theme_use())

        auto_size_label_rows(self)

        logging.info('Supported SAI versions:')
        for api in sorted(sai.sai_api_lookup.values(), key=lambda x: x.version_name):
            logging.info(f'  {api.version_name}')

        self.sai_recording_frame.set_button_callback(self.recording_wrapper(True, self.record_sai))
        self.psd_recording_frame.set_button_callback(self.recording_wrapper(True, self.record_psd))
        self.screen_click_button.set_callback(self.recording_wrapper(True, self.record_screen, grab=False))
        self.screen_grab_button.set_callback(self.recording_wrapper(True, self.record_screen, grab=True))
        self.export_config_frame.set_button_callback(self.recording_wrapper(False, self.export))

        self.sai_proc = None
        self.background_task = asyncio.create_task(self.run_background_task())
        self.operation_task: asyncio.Task | None = None

    def cleanup(self):
        self.win_size_var.set((self.winfo_width(), self.winfo_height()))
        self.settings.save()
        self.thread_running = False
        for task in [self.background_task, self.operation_task]:
            if task is not None:
                task.cancel()

    async def run_background_task(self):
        last_pid = None
        while True:
            if self.sai_proc is not None and self.sai_proc.is_running():
                canvases = [f'{canvas.get_name()} ({canvas.get_short_path()})' for canvas in self.sai_proc.get_canvas_list()]
                self.sai_canvas_box.set_values(canvases, init=True)
            else:
                self.sai_proc = None
                sai_pid = sai.find_running_sai_pid()
                if sai_pid is None:
                    self.sai_proc = None
                    self.sai_version_status_var.set("SAI is not running")
                elif sai_pid != last_pid:
                    last_pid = sai_pid
                    self.sai_proc = sai.SAI()
                    if self.sai_proc.is_sai_version_compatible(log=False):
                        self.sai_version_status_var.set(self.sai_proc.api.version_name)
                        continue
                    else:
                        self.sai_proc = None
                        self.sai_version_status_var.set("Unknown version")
            await asyncio.sleep(0.5)

    def run_operation(self, recording_mode, task, *args, **kwargs):
        if self.operation_task is not None:
            self.operation_task.cancel()
            self.operation_task = None
            self.thread_running = False
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
        frames_path = self.frames_file_var.get()
        container, codec = self.sai_recording_frame.get_format()
        image_size_limit = self.image_size_limit_var.get()
        canvases = self.sai_proc.get_canvas_list()
        if len(canvases) == 0:
            logging.info("There are no canvases open")
            return
        canvas = canvases[self.sai_canvas_box.get_index()]
        await timelapse.sai_capture(self.sai_proc, canvas, image_size_limit, frames_path, container, codec)

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
        if self.export_use_recording_var.get():
            container, codec = self.sai_recording_frame.get_format()
        else:
            container, codec = self.export_config_frame.get_format()
        output_path = self.export_file_var.get()
        def progress_kill_check(iterable, unit):
            for it in self.export_progress.iterate(iterable, unit):
                yield it
                if not self.thread_running:
                    logging.info('Export cancelled')
                    return
            self.thread_running = False
        self.thread_running = True
        await asyncio.to_thread(timelapse.export, progress_kill_check, export_time_limit, frames_path, container, codec, output_path)

    def report_callback_exception(self, exc_type, exc_value, exc_tb):
        logging.exception("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

async def main():
    await App().async_main_loop()