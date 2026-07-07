import ctypes
import hashlib
import sys
from functools import reduce
from operator import mul

import psutil
from PyMemoryEditor import OpenProcess
from PyMemoryEditor.process import AbstractProcess
import numpy as np

from . import _

def trim_null(data):
    if '\0' in data:
        data = data[0:data.index('\0')]
    return data

def from_wide_str(data):
    return trim_null(bytes(data).decode('utf-16le'))

class RemotePointerBase(ctypes.Structure):
    def get(self, proc:AbstractProcess, index=0, count=1, buffer=None):
        dtype = self._type_
        dsize = ctypes.sizeof(dtype)
        offset = dsize * index
        address = self.value + offset
        if buffer is None:
            byte_size = dsize * count
            try:
                data = proc.read_bytes(address, byte_size)
                if count == 1:
                    result = self._type_.from_buffer_copy(data)
                else:
                    result = (self._type_ * count).from_buffer_copy(data)
            except:
                if count == 1:
                    result = self._type_()
                else:
                    result = (self._type_ * count)()
            if isinstance(result, ctypes._SimpleCData):
                return result.value
            return result
        else:
            try:
                proc.read_process_memory_into(address, buffer)
            except:
                pass
            return buffer

    def __bool__(self):
        return bool(self.value)

    def __eq__(self, other):
        return self.value == other.value

class RemotePointer32(RemotePointerBase):
    prefix = 'RP32'
    _fields_ = [('value', ctypes.c_uint32)]

class RemotePointer64(RemotePointerBase):
    prefix = 'RP'
    _fields_ = [('value', ctypes.c_uint64)]

_remote_pointer_cache = {}
def _RPOINTER(dtype, rpointer_class):
    name = f'{rpointer_class.prefix}_{dtype.__qualname__}'
    if name in _remote_pointer_cache:
        return _remote_pointer_cache[name]
    ptype = type(name, (rpointer_class,), { '_type_': dtype })
    _remote_pointer_cache[name] = ptype
    return ptype

def RPOINTER32(dtype):
    return _RPOINTER(dtype, RemotePointer32)

def RPOINTER(dtype):
    return _RPOINTER(dtype, RemotePointer64)

def get_region_data_by_name(proc:AbstractProcess, name:str) -> tuple[str, int] | tuple[None, None]:
    for region in proc.get_memory_regions():
        if name in region.path:
            return region.path, region.address
    return None, None

def get_base_address(proc:AbstractProcess) -> int | None:
    return get_region_data_by_name(proc, psutil.Process(proc.pid).name())[1]

sai_api_lookup = {}
def register_sai_api(api):
    sai_api_lookup[api.exe_hash] = api
    return api

def get_sai_api(exe_hash):
    return sai_api_lookup.get(exe_hash)

def get_sai_api_list():
    return sorted(sai_api_lookup.values(), key=lambda x: x.__name__)

pad_index = 0
def pad(size):
    global pad_index
    field = (f'__pad{pad_index}', ctypes.c_char * size)
    pad_index += 1
    return field

def offset_fields(fields):
    fields = sorted(fields, key=lambda x: x[0])
    result = []
    current_offset = 0
    for offset, name, type in fields:
        if offset > current_offset:
            result.append(pad(offset - current_offset))
            current_offset = offset
        elif offset < current_offset:
            raise ValueError(f'field \'{name}\' offset {hex(offset)} overlaps the previous field at {hex(current_offset)}')
        result.append((name, type))
        current_offset += ctypes.sizeof(type)
    return result

class SAI_API_Base:
    def __init__(self, proc:AbstractProcess):
        self.proc = proc
        self.base_address = get_base_address(proc)

    def collect_canvases(self, canvas_ptr):
        result = []
        while canvas_ptr:
            canvas = canvas_ptr.get(self.proc)
            canvas._self_ptr = canvas_ptr
            result.append(canvas)
            canvas_ptr = canvas.next_canvas
        return result

    def update_canvas(self, canvas) -> None:
        canvas._self_ptr.get(self.proc, buffer=canvas)

    def get_map_level_for_size(self, canvas, size:int) -> int:
        return 0

    def check_if_canvas_exists(self, canvas) -> bool:
        return any((c._self_ptr == canvas._self_ptr for c in self.get_canvas_list()))

class SAICanvasBase(ctypes.Structure):
    _pack_ = 1
    _layout_ = 'ms'

    def get_name(self):
        return from_wide_str(self.name)

    def get_short_path(self):
        return from_wide_str(self.short_path)

class SAIv1_API_Base(SAI_API_Base):
    process_name = 'sai.exe'
    map_count = 1

    class SAIPixelHeap(ctypes.Structure):
        _pack_ = 1
        _layout_ = 'ms'
        _fields_ = offset_fields([
            (0x4, 'stride_x', ctypes.c_int32),
            (0x8, 'stride_y', ctypes.c_int32),
            (0xc, 'data', RPOINTER32(ctypes.c_uint8)),
        ])

    class SAICanvas(SAICanvasBase):
        pass

    SAICanvas._fields_ = offset_fields([
            (0x004, 'next_canvas', RPOINTER32(SAICanvas)),
            (0x030, 'pixel_heap_level_1', RPOINTER32(SAIPixelHeap)),
            (0x114, 'lower_pad_x', ctypes.c_int32),
            (0x118, 'lower_pad_y', ctypes.c_int32),
            (0x124, 'width', ctypes.c_int32),
            (0x128, 'height', ctypes.c_int32),
            (0x12c, 'padded_width', ctypes.c_int32),
            (0x130, 'padded_height', ctypes.c_int32),
            (0x4d0, 'short_path', ctypes.c_char * 0x108),
            (0x5d8, 'name', ctypes.c_char * 0x108),
        ])

    class SAISession(ctypes.Structure):
        _pack_ = 1
        _layout_ = 'ms'

    SAISession._fields_ = offset_fields([
        (0x4c, 'canvas_list', RPOINTER32(SAICanvas)),
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = np.empty((0,), dtype=np.uint8)

    def get_canvas_list(self) -> list[SAICanvas]:
        canvas_ptr = RPOINTER32(RPOINTER32(self.SAISession))(self.base_address + self.session_offset).get(self.proc).get(self.proc).canvas_list
        return self.collect_canvases(canvas_ptr)

    def check_if_canvas_exists(self, canvas:SAICanvas) -> bool:
        return any((c._self_ptr == canvas._self_ptr for c in self.get_canvas_list()))

    def get_canvas_image(self, canvas:SAICanvas, map_level=0) -> np.ndarray:
        self.update_canvas(canvas)
        lower_pad_x = canvas.lower_pad_x
        lower_pad_y = canvas.lower_pad_y
        pixel_heap = canvas.pixel_heap_level_1.get(self.proc)
        image_shape = (canvas.padded_height, canvas.padded_width, 4)
        if self.image.shape != image_shape:
            self.image = np.empty(shape=image_shape, dtype=np.uint8)
        pixel_heap.data.get(self.proc, buffer=self.image)
        return self.image[lower_pad_y : canvas.height + lower_pad_y, lower_pad_x : canvas.width + lower_pad_x, :3]

@register_sai_api
class SAIv1_API_1_2_5(SAIv1_API_Base):
    version_name = 'SAI Ver.1.2.5 (32bit)'
    exe_hash = 'f4e7e00aa4c222d6253aa1f0a5f302c2'
    session_offset = 0x491a8c

@register_sai_api
class SAIv1_API_1_2_6_Beta_3(SAIv1_API_Base):
    version_name = 'SAI Ver.1.2.6-Beta.3 (32bit)'
    exe_hash = 'b693f2c4516c09d008e30827208be1e6'
    session_offset = 0x494bcc

class SAIv2_API_Base(SAI_API_Base):
    process_name = 'sai2.exe'
    map_count = 0xb

    class SAICanvasTileMap(ctypes.Structure):
        _pack_ = 1
        _layout_ = 'ms'

        _fields_ = offset_fields([
            # (0x00, 'allocator', RPOINTER(WINFUNCTYPE(None, int64_t, int512_t, int512_t)))
            (0x08, 'tree', RPOINTER(RPOINTER(RPOINTER(ctypes.c_uint8)))),
            (0x10, 'width', ctypes.c_int32),
            (0x14, 'height', ctypes.c_int32),
            (0x18, 'count_x', ctypes.c_int32),
            (0x1c, 'count_y', ctypes.c_int32),
        ])

    class SAICanvas(SAICanvasBase):
        pass

    SAICanvas._fields_ = offset_fields([
            (0x000, 'next_canvas', RPOINTER(SAICanvas)),
            (0x020, 'id', ctypes.c_int32),
            (0x028, 'tile_maps', RPOINTER(RPOINTER(SAICanvasTileMap)) * map_count),
            (0x250, 'width', ctypes.c_int32),
            (0x254, 'height', ctypes.c_int32),
            (0x2ac, 'name', ctypes.c_uint16 * 0x100),
            (0x4ac, 'short_path', ctypes.c_uint16 * 0x100)
        ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = np.empty((0,), dtype=np.uint8)
        tile_shape = (256, 256, 4)
        self.tile = np.empty(shape=tile_shape, dtype=np.uint8)

    def get_canvas_list(self) -> list[SAICanvas]:
        canvas_ptr = RPOINTER(RPOINTER(self.SAICanvas))(self.base_address + self.session_offset).get(self.proc)
        return self.collect_canvases(canvas_ptr)[1:]

    def check_if_canvas_exists(self, canvas:SAICanvas) -> bool:
        return any((c.id == canvas.id for c in self.get_canvas_list()))

    def get_map_level_for_size(self, canvas:SAICanvas, size:int) -> int:
        self.update_canvas(canvas)
        map_level = 0
        for i in range(self.map_count):
            tile_map = canvas.tile_maps[i].get(self.proc).get(self.proc)
            if tile_map.width < size and tile_map.height < size:
                break
            map_level = i
        return map_level

    def get_canvas_image(self, canvas:SAICanvas, map_level=0) -> np.ndarray:
        self.update_canvas(canvas)
        tile_map = canvas.tile_maps[map_level].get(self.proc).get(self.proc)
        tiles_y = tile_map.count_y
        tiles_x = tile_map.count_x
        image_shape = (tiles_y * 256, tiles_x * 256, 4)
        if self.image.shape != image_shape:
            self.image = np.empty(shape=image_shape, dtype=np.uint8)
        for y in range(tiles_y):
            for x in range(tiles_x):
                tile_map.tree.get(self.proc, y).get(self.proc, x).get(self.proc, buffer=self.tile)
                self.image[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256] = self.tile
        return self.image[0 : tile_map.height, 0 : tile_map.width, :3]

@register_sai_api
class SAIv2_API_2023_07_11(SAIv2_API_Base):
    version_name= 'SAI Ver.2 (64bit) Preview.2023.07.11'
    exe_hash = 'f8df4067657d811e6e48c37c5b0f8fc5'
    session_offset = 0x31af20

@register_sai_api
class SAIv2_API_2024_11_23(SAIv2_API_Base):
    version_name = 'SAI Ver.2 (64bit) Preview.2024.11.23'
    exe_hash = 'bd60d6750ef57668f9bc44eb98d992c4'
    session_offset = 0x322620

@register_sai_api
class SAIv2_API_2026_07_02b(SAIv2_API_Base):
    version_name = 'SAI Ver.2 (64bit) Preview.2026.07.02b'
    exe_hash = '6f3f351303cf3896f4c2977925c994c2'
    session_offset = 0x32eb20

@register_sai_api
class SAIv2_API_2026_07_02b_alpha(SAIv2_API_Base):
    version_name = 'SAI Ver.2 (64bit) Alpha.2026.07.06'
    exe_hash = 'a20936b506674ed0f867e1ede8875ffa'
    session_offset = 0x46b980

    class SAICanvas(SAICanvasBase):
        pass

    SAICanvas._fields_ = offset_fields([
            (0x000, 'next_canvas', RPOINTER(SAICanvas)),
            (0x008, 'prev_canvas', RPOINTER(SAICanvas)),
            (0x010, 'prev_user_canvas', RPOINTER(SAICanvas)),
            (0x018, 'next_user_canvas', RPOINTER(SAICanvas)),
            # A canvas ID was here, but seems to be removed now.
            (0x028, 'width', ctypes.c_int32),
            (0x02c, 'height', ctypes.c_int32),
            (0x180, 'tile_count_x', ctypes.c_uint32),
            (0x184, 'tile_count_y', ctypes.c_uint32),
            (0x048, 'tile_maps', RPOINTER(RPOINTER(SAIv2_API_Base.SAICanvasTileMap)) * SAIv2_API_Base.map_count),
            (0x8f8, 'name', ctypes.c_uint16 * 0x104),
            (0xb00, 'short_path', ctypes.c_uint16 * 0x104)
        ])

    def check_if_canvas_exists(self, canvas:SAICanvas) -> bool:
        return SAI_API_Base.check_if_canvas_exists(self, canvas)

def get_pid_by_name(name:str) -> int | None:
    psutil.process_iter.cache_clear()
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.name() == name:
            return proc.pid
    return None

def find_running_sai_pid() -> int | None:
    return get_pid_by_name(SAIv1_API_Base.process_name) or get_pid_by_name(SAIv2_API_Base.process_name)

def get_exe_path(proc:AbstractProcess) -> str:
    psproc = psutil.Process(proc.pid)
    if 'win' in sys.platform:
        return psproc.exe()
    else: # linux
        return get_region_data_by_name(proc, psproc.name())[0]

def get_exe_hash(proc:AbstractProcess) -> str | None:
    exe_path = get_exe_path(proc)
    if not exe_path:
        return None
    with open(exe_path, 'rb') as f:
        return hashlib.file_digest(f, 'md5').hexdigest()

class NoSaiProcessDetected(Exception):
    pass

def get_sai_api_from_pid(pid:int) -> SAI_API_Base | None:
    try:
        with OpenProcess(pid=pid) as proc:
            exe_hash = get_exe_hash(proc)
            return get_sai_api(exe_hash)
    except:
        raise NoSaiProcessDetected()

class SAI:
    def __init__(self, override_api:SAI_API_Base=None):
        self.proc = None
        pid = find_running_sai_pid()
        try:
            self.proc = OpenProcess(pid=pid)
            self.psutil_proc = psutil.Process(pid=pid)
        except:
            raise NoSaiProcessDetected()
        if override_api is None:
            api = get_sai_api_from_pid(pid)
        else:
            api = override_api
        self.api:SAI_API_Base = api(self.proc)

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    def close(self):
        if self.proc is not None:
            self.proc.close()

    def is_running(self):
        return self.psutil_proc.is_running()

    def get_pid(self):
        return self.proc.pid

def test():
    with SAI() as sai:
        canvas = sai.get_canvas_list()[0]
        print('Canvas name:', canvas.get_name())
        img = sai.get_canvas_image(canvas, 5)
        import cv2
        cv2.imwrite('test.jpg', img)