import asyncio
import logging
from pathlib import Path
import json
from collections import deque
import os
import threading
import traceback
import time

import tkinter as tk
import ttkbootstrap as ttk
import ttkbootstrap.constants as ttkc
from ttkbootstrap.widgets import tooltip, scrolled
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

from . import asynctk, timelapse, sai, set_locale, _ as tr

def _(t):
    return t

translatables = []

def get_fixed_font(font_name='Consolas', font_size=10):
    if font_name in tk.font.families():
        font = (font_name, font_size)
    else:
        if font_name not in ttk.font.names():
            font_name = 'TkFixedFont'
        font = tk.font.nametofont(font_name)
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

class LocalizedStringVar(ttk.StringVar):
    def __init__(self, text, *args, **kwargs):
        super().__init__(*args, **kwargs)
        translatables.append(self)
        self.set(text)

    def set(self, text):
        self.init_text = text
        self.update_translation()

    def update_translation(self):
        super().set(tr(self.init_text))

class StackStringVar(ttk.StringVar):
    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = []
        self.cbname = None
        self.linked_var = None
        self.push(value)

    def unset_value(self):
        if self.linked_var is not None:
            self.linked_var.trace_remove('write', self.cbname)
            self.linked_var = None
            self.cbname = None

    def set_value(self, value):
        self.unset_value()
        if isinstance(value, ttk.StringVar):
            self.cbname = value.trace_add('write', lambda *_: self.set(value.get()))
            self.linked_var = value
            self.set(value.get())
        else:
            self.set(value)

    def push(self, value):
        self.set_value(value)
        self.stack.append(value)

    def pop(self):
        value = self.stack.pop()
        if self.stack:
            self.set_value(self.stack[-1])
        else:
            self.set_value('')

class ButtonRow(ttk.Frame):
    def __init__(self, master, text=None, textvariable=None):
        super().__init__(master)
        self.pack(fill=ttkc.X)
        self.button = ttk.Button(self, text=text, textvariable=textvariable)
        self.button.pack(side=ttkc.LEFT, fill=ttkc.X, expand=True)

    def set_text(self, text):
        self.button.config(text=text)

    def set_callback(self, callback):
        self.button.config(command=callback)

class ProgressRow(ttk.Frame):
    def __init__(self, master, font=None):
        super().__init__(master)
        self.pack(fill=ttkc.X)
        self.label = ttk.Label(self, font=font)
        self.label.pack()
        self.progressbar = ttk.Progressbar(self)
        self.progressbar.pack(fill=ttkc.X, expand=True)
        self.loop = asyncio.get_running_loop()

    def iterate(self, iterable, total, unit='it'):
        self.last_update = float('-inf')
        self.avg_ips = RollingAverage()
        self.avg_eta_sec = RollingAverage()
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
    def __init__(self, master, label):
        super().__init__(master)
        self.pack(fill=ttkc.X)
        self.label = ttk.Label(self, text=label, textvariable=LocalizedStringVar(label))
        self.label.pack(side=ttkc.LEFT)

class StatusLabelRow(LabelRow):
    def __init__(self, master, label_text, status_text='', textvariable=None):
        super().__init__(master, label_text)
        self.status = ttk.Entry(self, text=status_text, textvariable=textvariable, state=ttkc.READONLY)
        self.status.pack(side=ttkc.LEFT, fill=ttkc.X, expand=True)

class ImageLabelRow(LabelRow):
    def __init__(self, master, label):
        super().__init__(master, label)
        self.canvas = ttk.Canvas(self, height=100, width=100)
        self.canvas.pack(side=ttkc.LEFT)
        self.image_tag = None

    def set_image(self, image):
        self.image = ImageTk.PhotoImage(image)
        if self.image_tag is None:
            self.image_tag = self.canvas.create_image(50, 50, anchor=ttkc.CENTER, image=self.image)
        self.canvas.itemconfig(self.image_tag, image=self.image)

    def remove_image(self):
        if self.image_tag is not None:
            self.canvas.delete(self.image_tag)
            self.image_tag = None

class CheckbuttonLabelRow(LabelRow):
    def __init__(self, master, label, variable=None):
        super().__init__(master, label)
        self.checkbutton = ttk.Checkbutton(self, variable=variable)
        self.checkbutton.pack(side=ttkc.LEFT)

def check_digit(p):
    return str(p).isdigit() or str(p) == ''

class EntryLabelRow(LabelRow):
    def __init__(self, master, label, numbers=False, button_label=None, textvariable=None):
        super().__init__(master, label)
        self.textvariable = textvariable or ttk.StringVar()
        kwargs = {}
        if numbers:
            kwargs['validate'] = ttkc.ALL
            proc = self.register(check_digit)
            kwargs['validatecommand'] = (proc, '%P')
        kwargs['textvariable'] = self.textvariable
        self.entry = ttk.Entry(self, **kwargs)
        self.entry.pack(side=ttkc.LEFT, fill=ttkc.X, expand=True)
        self.button = None
        if button_label is not None:
            self.button = ttk.Button(self, text=button_label, width=10, textvariable=LocalizedStringVar(button_label))
            self.button.pack(side=ttkc.LEFT)

    def enable(self, state):
        self.entry.config(state=ttkc.NORMAL if state else ttkc.DISABLED)

def get_nearest_dir(path):
    path = timelapse.expand_path(path).absolute()
    if path.is_dir():
        return path
    for p in path.parents:
        if p.is_dir():
            return p
    return Path('.')

class FilePickerEntry(EntryLabelRow):
    def __init__(self, master, text, title=_('Select'), mode='open', filetypes=((_('Any'), '*.*'),), button_label=_('Select'), **kwargs):
        super().__init__(master, text, button_label=button_label, **kwargs)
        self.title = title
        self.mode = mode
        self.filetypes = filetypes
        self.button.config(command=self.pick_file)

    def pick_file(self, *_args):
        initdir = get_nearest_dir(self.textvariable.get())
        title = tr(self.title)
        filetypes = [tuple((tr(x) for x in ftype)) for ftype in self.filetypes]
        filename = None
        match self.mode:
            case 'open':
                filename = filedialog.askopenfilename(
                    title=title,
                    initialdir=initdir,
                    filetypes=filetypes,
                )
            case 'save':
                filename = filedialog.asksaveasfilename(
                    title=title,
                    initialdir=initdir,
                    filetypes=filetypes,
                )
            case 'dir':
                filename = filedialog.askdirectory(
                    title=title,
                    initialdir=initdir,
                )
            case _:
                return
        if filename is not None and filename:
            self.textvariable.set(filename)

class ComboboxLabelRow(LabelRow):
    def __init__(self, master, text, values=None, index_variable=None):
        super().__init__(master, text)
        self.index_var = index_variable
        if self.index_var is None:
            self.index_var = ttk.IntVar(value=0)
        self.combobox = ttk.Combobox(self, state=ttkc.READONLY)
        self.combobox.pack(side=ttkc.LEFT, fill=ttkc.X, expand=True)
        self.reentering = False
        self.combobox.bind('<<ComboboxSelected>>', self.on_selected)
        self.index_var.trace_add('write', self.var_trace)
        self.values = values
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
        self.combobox.config(values=[value.get() if isinstance(value, ttk.StringVar) else value for value in values])
        self.values = values
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

    def get_values(self):
        return self.values

class ToolTip(tooltip.ToolTip):
    def __init__(self, widget, text, *args, **kwargs):
        super().__init__(widget, text, *args, **kwargs)
        self.textvariable = LocalizedStringVar(text)
        self.textvariable.trace_add('write', self.update_text)
        self.update_text()

    def update_text(self, *_args):
        self.text = self.textvariable.get()

class VideoConfigFrame(ttk.Frame):
    def __init__(self, master, shared_vars):
        translatables.append(self)
        super().__init__(master)
        self.pack(fill=ttkc.X)
        self.video_types = [
            (_('mp4/avc1 - Works with most websites, including Twitter'), ('mp4', 'avc1')),
            (_('webm/vp80 - Better color quality, but larger file size'), ('webm', 'vp80')),
            (_('Use custom container/codec'), None)
        ]
        self.video_type_var = shared_vars['video_type_var']
        self.custom_container_var = shared_vars['custom_container_var']
        self.custom_codec_var = shared_vars['custom_codec_var']
        self.button_text = shared_vars['button_text']
        self.video_type_box = ComboboxLabelRow(self, _('Video type'), values=[t[0] for t in self.video_types], index_variable=self.video_type_var)
        ToolTip(self.video_type_box, _('Useful container/codec presets.'))
        self.custom_container_entry = EntryLabelRow(self, _('Custom container'), textvariable=self.custom_container_var)
        ToolTip(self.custom_container_entry, _('If you want to use a container that is not listed above, whatever is supported by OpenCV and FFMPEG.'))
        self.custom_codec_entry = EntryLabelRow(self, _('Custom codec'), textvariable=self.custom_codec_var)
        ToolTip(self.custom_codec_entry, _('If you want to use a codec that is not listed above, whatever is supported by OpenCV and FFMPEG.'))
        self.button = ButtonRow(self, textvariable=self.button_text)

    def get_format(self):
        video_format = self.video_types[self.video_type_var.get()][1]
        if video_format is None:
            video_format = self.custom_container_var.get() or 'mp4', self.custom_codec_var.get() or 'avc1'
        return video_format

    def set_button_callback(self, callback):
        self.button.set_callback(callback)

    def update_translation(self):
        values = [tr(vtype[0]) for vtype in self.video_types]
        self.video_type_box.set_values(values)

class StatusArea(ttk.Labelframe):
    def __init__(self, master, font=None):
        super().__init__(master, text=_('Status'))
        self.pack(fill=ttkc.BOTH, expand=True)
        self.text = scrolled.ScrolledText(self, font=font, width=1, height=1)
        self.text.text.config(state=ttkc.DISABLED)
        self.text.pack(fill=ttkc.BOTH, expand=True)

    def append(self, text):
        self.text.text.config(state=ttkc.NORMAL)
        self.text.insert(index=ttkc.END, chars=f'{text}\n')
        self.text.see(ttkc.END)
        self.text.text.config(state=ttkc.DISABLED)

class AsyncWidgetLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        self.widget = widget
        self.loop = asyncio.get_running_loop()

    def emit(self, record):
        self.loop.call_soon_threadsafe(self.widget.append, self.format(record))

def make_frames_path_field(master, textvariable=None):
    frames_path = FilePickerEntry(master, _('Frames path'), mode='dir', textvariable=textvariable)
    ToolTip(frames_path, _('Folder where video recording cuts are saved.'))

def make_auto_split_field(master, textvariable=None):
    auto_split = EntryLabelRow(master, _('Auto split count'), numbers=True, textvariable=textvariable)
    ToolTip(auto_split, _('Automatically split video output after this many frames. Enter 0 or leave blank to disable. This is useful to mitigate the chance of missing or corrupt video data after a system crash.'))

def make_image_size_limit_box(master, textvariable=None):
    frame = EntryLabelRow(master, _('Image size limit (px)'), numbers=True, textvariable=textvariable)
    ToolTip(frame, _('Captured images will be resized to this if larger than this value.'))
    return frame

def make_notebook_frame(notebook:ttk.Notebook, text:str):
    frame = ttk.Frame(notebook)
    frame.pack(fill=ttkc.BOTH, expand=True)
    notebook.add(frame, text=text)
    var = LocalizedStringVar(text)
    tab_id = notebook.index('end') - 1
    var.trace_add('write', lambda *_args: notebook.tab(tab_id, text=var.get()))
    frame._tab_name_var = var
    return frame

def get_widgets_by_type(master, wtype):
    for widget in master.winfo_children():
        if isinstance(widget, wtype):
            yield widget
        yield from get_widgets_by_type(widget, wtype)

def get_text_width(label):
    return tk.font.Font(font=label['font']).measure(label['text'])

def auto_size_label_rows(master):
    label_rows = list(get_widgets_by_type(master, LabelRow))
    max_width = 0
    for row in label_rows:
        row.label.config(width=0)
        max_width = max(max_width, row.label.winfo_reqwidth())
    for row in label_rows:
        row.label.pack(side=ttkc.LEFT, ipadx=max_width/2, ipady=6, padx=5, fill=ttkc.X)
        row.label.config(width=1)

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
        self.title(f'art-timelapse {timelapse.__version__}')
        self.settings = Settings('.art-timelapse')
        fixed_font = get_fixed_font()

        #########################################################
        # Tk variables
        self.win_size_var = self.settings.get_var('win_size', (600, 600))
        self.frames_file_var = self.settings.get_var('frames_file', '')
        self.auto_split_sai_var = self.settings.get_var('auto_split_sai', 500)
        self.auto_split_psd_var = self.settings.get_var('auto_split_psd', 10)
        self.auto_split_screen_var = self.settings.get_var('auto_split_screen', 500)
        self.recording_button_text_var = StackStringVar(value=LocalizedStringVar(_('Start Recording')))
        self.export_button_text_var = StackStringVar(value=LocalizedStringVar(_('Export')))
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
        self.sai_version_status_var = LocalizedStringVar('')
        self.sai_version_override_var = self.settings.get_var('sai_version_override', 0)
        self.sai_canvas_var = self.settings.get_var('sai_canvas', '')
        self.image_size_limit_var = self.settings.get_var('image_size_limit', 1000)
        self.psd_file_var = self.settings.get_var('psd_file_var', '')
        self.screen_click_button_var = StackStringVar(value=LocalizedStringVar(_('Click window grab and start recording')))
        self.screen_grab_button_var = StackStringVar(value=LocalizedStringVar(_('Drag area grab and start recording')))
        self.export_file_var = self.settings.get_var('export_file_var', '')
        self.export_time_limit_var = self.settings.get_var('export_time_limit', 60)
        self.export_fps_var = self.settings.get_var('export_fps', 30)
        self.export_use_recording_var = self.settings.get_var('export_user_recording', True)
        self.selected_tab_var = self.settings.get_var('selected_tab', 0)
        self.selected_tab_thread_safe = 0
        meta_config_theme_var = self.settings.get_var('theme', 'darkly')
        meta_config_lang_var = self.settings.get_var('lang', 'en')

        #########################################################
        # GUI widgets and layout
        notebook = ttk.Notebook(self)
        notebook.pack(side=ttkc.TOP, fill=ttkc.X)

        sai_frame = make_notebook_frame(notebook, _('SAI Recording'))
        make_frames_path_field(sai_frame, self.frames_file_var)
        make_auto_split_field(sai_frame, self.auto_split_sai_var)
        StatusLabelRow(sai_frame, _('SAI version detected'), textvariable=self.sai_version_status_var)
        self.sai_version_override_box = ComboboxLabelRow(sai_frame, _('SAI version override'), values=[api.version_name for api in sai.get_sai_api_list()], index_variable=self.sai_version_override_var)
        self.sai_canvas_box = ComboboxLabelRow(sai_frame, _('Canvas'), index_variable=self.sai_canvas_var)
        self.sai_canvas_preview = ImageLabelRow(sai_frame, _('Canvas preview'))
        make_image_size_limit_box(sai_frame, textvariable=self.image_size_limit_var)
        self.sai_recording_frame = VideoConfigFrame(sai_frame, recording_vars)

        psd_frame = make_notebook_frame(notebook, _('PSD Recording'))
        make_frames_path_field(psd_frame, self.frames_file_var)
        make_auto_split_field(psd_frame, self.auto_split_psd_var)
        psd_file_entry = FilePickerEntry(psd_frame, _('PSD file'), mode='open', filetypes=[('PSD', '.psd .psb'), (_('Any'), '*.*')], textvariable=self.psd_file_var)
        make_image_size_limit_box(psd_frame, textvariable=self.image_size_limit_var)
        self.psd_recording_frame = VideoConfigFrame(psd_frame, recording_vars)

        screen_frame = make_notebook_frame(notebook, _('Screen Recording'))
        make_frames_path_field(screen_frame, self.frames_file_var)
        make_auto_split_field(screen_frame, self.auto_split_screen_var)
        self.screen_recording_frame = VideoConfigFrame(screen_frame, recording_vars)
        self.screen_recording_frame.button.forget()
        self.screen_click_button = ButtonRow(screen_frame, textvariable=self.screen_click_button_var)
        self.screen_grab_button = ButtonRow(screen_frame, textvariable=self.screen_grab_button_var)

        export_frame = make_notebook_frame(notebook, _('Export Video'))
        make_frames_path_field(export_frame, self.frames_file_var)
        export_file_entry = FilePickerEntry(export_frame, _('Export file'), mode='save', textvariable=self.export_file_var)
        export_time_limit_entry = EntryLabelRow(export_frame, _('Export time limit'), numbers=True, textvariable=self.export_time_limit_var)
        export_fps_entry = EntryLabelRow(export_frame, _('Export FPS'), numbers=True, textvariable=self.export_fps_var)
        export_user_reocrding = CheckbuttonLabelRow(export_frame, _('Use recording video type'), variable=self.export_use_recording_var)
        self.export_config_frame = VideoConfigFrame(export_frame, export_vars)
        self.export_progress = ProgressRow(export_frame, font=fixed_font)

        status_area = StatusArea(self, font=fixed_font)
        logger = AsyncWidgetLogger(status_area)
        logging.getLogger().addHandler(logger)

        self.stop_recording_label = LocalizedStringVar(_('Stop Recording'))
        self.stop_exporting_label = LocalizedStringVar(_('Stop Exporting'))

        ToolTip(self.sai_version_override_box, _('If the running SAI version cannot be detected, the selected override will be used.'))
        ToolTip(self.sai_canvas_box, _('Select which open SAI canvas to record from.'))
        ToolTip(psd_file_entry, _('PSD/PSB file to record from as it is saved to disk.'))
        ToolTip(self.screen_click_button, _('Captures the subwindow that was clicked on. Automatically adjusts capture size to the subwindow.'))
        ToolTip(self.screen_grab_button, _('Captures a fixed area of the subwindow. Drag a rectangle like a screenshot tool.'))
        ToolTip(export_file_entry, _('Path to the video file to export to. Extension is determined by video type. Leave blank to automatically use the frames path.'))
        ToolTip(export_time_limit_entry, _('Maximum time the exported video should be. Enter 0 or leave blank for no time limit.'))
        ToolTip(export_fps_entry, _('Set the FPS of the exported video. 30 is a common default. Lower FPS is better for PSD recording exports (like 5 FPS).'))
        ToolTip(export_user_reocrding, _('Use the same video options as the recording tabs and ignore the below settings.'))

        meta_config = ttk.Frame(self)
        meta_config.pack(fill=ttkc.X)
        theme_label = LocalizedStringVar(_('Theme:'))
        ttk.Label(meta_config, textvariable=theme_label).pack(side=ttkc.LEFT)
        meta_config_theme_box = ttk.Combobox(meta_config, state=ttkc.READONLY)
        meta_config_theme_box.pack(side=ttkc.LEFT)
        meta_config_theme_box.config(values=ttk.Style().theme_names(), textvariable=meta_config_theme_var)

        meta_config_lang_box = ttk.Combobox(meta_config, state=ttkc.READONLY)
        meta_config_lang_box.pack(side=ttkc.RIGHT)
        ttk.Label(meta_config, text='🌐').pack(side=ttkc.RIGHT)
        languages_lookup = {
            'English': 'en',
            '日本語': 'ja',
        }
        languages_lookup_rev = { v:k for k,v in languages_lookup.items() }
        lang_box_var = ttk.StringVar()
        info_lang_printed = set()
        def on_lang_change(*_args):
            lang_code = languages_lookup.get(lang_box_var.get())
            set_locale(lang_code)
            meta_config_lang_var.set(lang_code)
            for obj in translatables:
                obj.update_translation()
            auto_size_label_rows(self)
            if lang_code not in info_lang_printed:
                info_lang_printed.add(lang_code)
                self.print_init_info()
        lang_box_var.trace_add('write', on_lang_change)
        meta_config_lang_box.config(values=list(languages_lookup.keys()), textvariable=lang_box_var)
        meta_config_lang_box.set(languages_lookup_rev.get(meta_config_lang_var.get()))

        #########################################################
        # Init and callbacks
        self.geometry('{}x{}'.format(*self.win_size_var.get()))

        def on_notebook_tab_changed(*_args):
            index = notebook.index(notebook.select())
            self.selected_tab_thread_safe = index
            self.selected_tab_var.set(index)
            notebook.config(height=notebook.winfo_children()[index].winfo_reqheight())
        notebook.bind('<<NotebookTabChanged>>', on_notebook_tab_changed)
        notebook.select(notebook.tabs()[self.selected_tab_var.get()])
        def hack_fix_size():
            if notebook.winfo_height() <= 1:
                on_notebook_tab_changed()
                self.after(10, hack_fix_size)
        self.after(10, hack_fix_size)

        meta_config_theme_var.trace_add('write', lambda *_: ttk.Style(meta_config_theme_var.get()))
        meta_config_theme_var.set(meta_config_theme_var.get())

        self.sai_proc = None
        self.last_pid = None
        self.current_pid = None

        self.sai_version_override_var.trace_add('write', self.on_version_override_changed)
        self.sai_canvas_var.trace_add('write', self.on_canvas_selected)
        self.bind('<FocusIn>', self.on_canvas_selected)
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
        self.canvas_preview_task: asyncio.Task | None = None

    def print_init_info(self):
        session_type_key = 'XDG_SESSION_TYPE'
        session_type = os.environ.get(session_type_key, '')
        if session_type.lower() == 'wayland':
            logging.warning(tr(_('Possible Wayland session is running because the following environment variable was set:')))
            logging.warning(f'  {session_type_key}={session_type}')
            logging.warning(tr(_('Wayland sessions are currently not supported by dependencies mss and pywinctl.')))
            logging.warning(tr(_('To work around this, you can run this tool and your desired art program inside of Xwayland, see README.md.')))

        logging.info(tr(_('Supported SAI versions:')))
        for api in sorted(sai.sai_api_lookup.values(), key=lambda x: x.version_name):
            logging.info(f'  {api.version_name}')

    def cleanup(self):
        self.win_size_var.set((self.winfo_width(), self.winfo_height()))
        self.settings.save()
        self.pid_scan_event.set()
        self.operation_thread_running = False
        for task in [self.background_task, self.operation_task, self.canvas_preview_task]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    ex = task.exception()
                    logging.exception(ex)
                except:
                    pass

    def on_version_override_changed(self, *_args):
        self.clear_sai_proc()
        self.set_sai_proc()

    def on_canvas_selected(self, *_args):
        self.update_canvas_preview()

    def update_canvas_preview(self, load_wait=False):
        if self.sai_proc is None:
            self.sai_canvas_preview.remove_image()
            return
        if self.canvas_preview_task is not None:
            self.canvas_preview_task.cancel()
        async def task():
            canvases = self.sai_proc.api.get_canvas_list()
            if not canvases:
                self.sai_canvas_preview.remove_image()
                return
            target_canvas = canvases[self.sai_canvas_box.get_index()]
            def canvas_get():
                if load_wait:
                    time.sleep(1.0)
                map_level = self.sai_proc.api.get_map_level_for_size(target_canvas, 100)
                image = self.sai_proc.api.get_canvas_image(target_canvas, map_level)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image.thumbnail((100, 100))
                return image
            image = await asyncio.to_thread(canvas_get)
            self.sai_canvas_preview.set_image(image)
        self.canvas_preview_task = asyncio.create_task(task())

    def clear_sai_proc(self):
        if self.sai_proc is not None:
            self.sai_proc.close()
            self.sai_proc = None
            self.last_pid = None

    def refresh_sai_canvses(self):
        canvases = [f'{canvas.get_name()} ({canvas.get_short_path()})' for canvas in self.sai_proc.api.get_canvas_list()]
        previous_canvases = self.sai_canvas_box.get_values()
        self.sai_canvas_box.set_values(canvases)
        if canvases != previous_canvases:
            self.update_canvas_preview(load_wait=True)

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
                self.sai_version_status_var.set(_('SAI is not running'))
            elif sai_pid != self.last_pid:
                try:
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
                        self.sai_version_status_var.set(f'{self.sai_proc.api.version_name}' + tr(_('(Override)')))
                    self.refresh_sai_canvses()
                except sai.NoSaiProcessDetected:
                    pass

    async def run_background_task(self):
        while True:
            if self.operation_task is None and self.selected_tab_var.get() == 0:
                try:
                    self.set_sai_proc()
                except Exception as ex:
                    logging.exception(ex)
            await asyncio.sleep(0.25)

    def run_operation(self, recording_mode, task, *args, **kwargs):
        if self.operation_task is not None:
            self.operation_task.cancel()
            self.operation_task = None
            self.operation_thread_running = False
            return
        async def subtask():
            if recording_mode:
                start_log = _('Recording started')
                stop_log = _('Recording stopped')
                stop_text = self.stop_recording_label
            else:
                start_log = _('Exporting started')
                stop_log = _('Exporting stopped')
                stop_text = self.stop_exporting_label
            logging.info(tr(start_log))
            try:
                svars = [self.recording_button_text_var, self.export_button_text_var,
                    self.screen_grab_button_var, self.screen_click_button_var]
                for var in svars:
                    var.push(stop_text)
                await task(*args, **kwargs)
            except asyncio.CancelledError:
                pass
            except Exception as ex:
                logging.exception(ex)
            finally:
                logging.info(tr(stop_log))
                for var in svars:
                    var.pop()
                self.operation_task = None
        self.operation_task = asyncio.create_task(subtask())

    def recording_wrapper(self, recording_mode, task, *args, **kwargs):
        return lambda: self.run_operation(recording_mode, task, *args, **kwargs)

    async def record_sai(self):
        if self.sai_proc is None:
            logging.info(tr(_('No valid SAI process found')))
            return
        with sai.SAI(type(self.sai_proc.api)) as sai_proc:
            canvases = sai_proc.api.get_canvas_list()
            if not canvases:
                logging.info(tr(_('There are no canvases open')))
                return
            canvas = canvases[self.sai_canvas_box.get_index()]
            frames_path = self.frames_file_var.get()
            container, codec = self.sai_recording_frame.get_format()
            image_size_limit = self.image_size_limit_var.get()
            auto_split = self.auto_split_sai_var.get()
            await timelapse.sai_capture(sai_proc, canvas, image_size_limit, frames_path, container, codec, auto_split)

    async def record_psd(self):
        frames_path = self.frames_file_var.get()
        container, codec = self.sai_recording_frame.get_format()
        image_size_limit = self.image_size_limit_var.get()
        psd_file = self.psd_file_var.get()
        auto_split = self.auto_split_psd_var.get()
        await timelapse.psd_capture(psd_file, image_size_limit, frames_path, container, codec, auto_split)

    async def record_screen(self, grab):
        window, bbox = await timelapse.get_window_and_bbox(grab)
        if window is None:
            return
        frames_path = self.frames_file_var.get()
        container, codec = self.sai_recording_frame.get_format()
        auto_split = self.auto_split_screen_var.get()
        await timelapse.screen_capture(window, bbox, frames_path, container, codec, auto_split)

    async def export(self):
        frames_path = self.frames_file_var.get()
        export_time_limit = self.export_time_limit_var.get()
        export_fps = self.export_fps_var.get()
        if self.export_use_recording_var.get():
            container, codec = self.sai_recording_frame.get_format()
        else:
            container, codec = self.export_config_frame.get_format()
        output_path = self.export_file_var.get()
        def progress_kill_check(iterable, total, unit):
            for it in self.export_progress.iterate(iterable, total, unit):
                yield it
                if not self.operation_thread_running:
                    logging.info(tr(_('Export cancelled')))
                    break
        self.operation_thread_running = True
        # Export can have long CPU-bound sections, so run it on another thread.
        await asyncio.to_thread(timelapse.export, progress_kill_check, export_time_limit, export_fps, frames_path, container, codec, output_path)

    def report_callback_exception(self, exc_type, exc_value, exc_tb):
        logging.exception(''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))

async def async_main():
    await App().async_main_loop()

def main():
    asyncio.run(async_main())