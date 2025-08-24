import ctypes
import hashlib
import sys
from functools import reduce
from operator import mul

import psutil
from PyMemoryEditor import OpenProcess
from PyMemoryEditor.process import util
import numpy as np

def from_wide_str(data):
    return bytes(data).decode('utf-16le')

class RemotePointerBase(ctypes.Structure):
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        unpack = lambda p, i=0, s=1:(p, i, s)
        proc, index, count = unpack(*key)
        dtype = self._type_
        dsize = ctypes.sizeof(dtype)
        offset = dsize * index
        byte_size = dsize * count
        address = self.value + offset
        data = proc.read_process_memory(address, bytes, byte_size)
        if count == 1:
            result = self._type_.from_buffer_copy(data)
        else:
            result = (self._type_ * count).from_buffer_copy(data)
        if isinstance(result, ctypes._SimpleCData):
            return result.value
        return result

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

if 'win' in sys.platform:
    psapi = ctypes.WinDLL('Psapi.dll')
    k32 = ctypes.windll.kernel32

    def get_base_address(pid):
        PROCESS_ALL_ACCESS = 0x1f0fff
        process_handle = k32.OpenProcess(
            PROCESS_ALL_ACCESS,
            False,
            pid
        )
        modules = (ctypes.c_void_p * 1)()
        LIST_MODULES_ALL = 0x3
        psapi.EnumProcessModulesEx(
            process_handle,
            ctypes.byref(modules),
            ctypes.sizeof(modules),
            ctypes.byref(ctypes.c_ulong()),
            LIST_MODULES_ALL
        )
        k32.CloseHandle(process_handle)
        return modules[0]

sai_api_lookup = {}
def add_sai_api(api):
    sai_api_lookup[api.exe_hash] = api

def get_sai_api(hash):
    result = sai_api_lookup.get(hash)
    if result is None:
        result = sorted(sai_api_lookup.values(), key=lambda x: x.__name__)[-1]
    return result

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

class SAI_API_Base():
    def __init__(self, proc, base_address):
        self.proc = proc
        self.base_address = base_address

    def collect_canvases(self, canvas_ptr):
        result = []
        while canvas_ptr:
            canvas = canvas_ptr[self.proc]
            canvas._self_ptr = canvas_ptr
            result.append(canvas)
            canvas_ptr = canvas.next_canvas
        return result

class SAIv1_API_Base(SAI_API_Base):
    process_name = 'sai.exe'

    def get_canvas_list(self):
        canvas_ptr = RPOINTER32(RPOINTER32(self.SAISession))(self.base_address + self.session_offset)[self.proc][self.proc].canvas_list
        return self.collect_canvases(canvas_ptr)

    def check_if_canvas_exists(self, canvas):
        return any((c._self_ptr == canvas._self_ptr for c in self.get_canvas_list()))

    def get_canvas_image(self, canvas):
        lower_pad_x = canvas.lower_pad_x
        lower_pad_y = canvas.lower_pad_y
        pixel_heap = canvas.pixel_heap_level_1[self.proc]
        data_shape = (canvas.padded_height, canvas.padded_width, 4)
        data_size = reduce(mul, data_shape)
        data = pixel_heap.data[self.proc, 0, data_size]
        image = np.frombuffer(data, dtype=np.uint8).reshape(data_shape)
        return image[lower_pad_y : canvas.height + lower_pad_y, lower_pad_x : canvas.width + lower_pad_x, :3]

class SAIv1_API_1_2_5(SAIv1_API_Base):
    version_name = 'SAI Ver.1.2.5 (32bit)'
    exe_hash = 'f4e7e00aa4c222d6253aa1f0a5f302c2'
    session_offset = 0x491a8c

    class SAIPixelHeap(ctypes.Structure):
        _pack_ = 1
        _fields_ = offset_fields([
            (0x4, 'stride_x', ctypes.c_int32),
            (0x8, 'stride_y', ctypes.c_int32),
            (0xc, 'data', RPOINTER32(ctypes.c_uint8)),
        ])

    class SAICanvas(ctypes.Structure):
        _pack_ = 1

        def get_name(self):
            return self.name.decode()

        def get_short_path(self):
            return self.short_path.decode()

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
    SAISession._fields_ = offset_fields([
        (0x4c, 'canvas_list', RPOINTER32(SAICanvas)),
    ])
add_sai_api(SAIv1_API_1_2_5)

class SAIv1_API_1_2_6_Beta_3(SAIv1_API_1_2_5):
    version_name = 'SAI Ver.1.2.6-Beta.3 (32bit)'
    exe_hash = 'b693f2c4516c09d008e30827208be1e6'
    session_offset = 0x494bcc
add_sai_api(SAIv1_API_1_2_6_Beta_3)

class SAIv2_API_Base(SAI_API_Base):
    process_name = 'sai2.exe'

    def get_canvas_list(self):
        canvas_ptr = RPOINTER(RPOINTER(self.SAICanvas))(self.base_address + self.canvas_list_offset)[self.proc]
        return self.collect_canvases(canvas_ptr)[1:]

    def check_if_canvas_exists(self, canvas):
        return any((c.id == canvas.id for c in self.get_canvas_list()))

    def get_canvas_image(self, canvas):
        tile_map = canvas.tile_map[self.proc][self.proc]
        tiles_y = tile_map.count_y
        tiles_x = tile_map.count_x
        image = np.empty(shape=(tiles_y * 256, tiles_x * 256, 4), dtype=np.uint8)
        tile_shape = (256, 256, 4)
        tile_size = reduce(mul, tile_shape)
        for y in range(tiles_y):
            for x in range(tiles_x):
                tile = tile_map.tree[self.proc, y][self.proc, x][self.proc, 0, tile_size]
                tile = np.frombuffer(tile, dtype=np.uint8).reshape(tile_shape)
                image[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256] = tile
        return image[0 : canvas.height, 0 : canvas.width, :3]

class SAIv2_API_2024_11_23(SAIv2_API_Base):
    version_name = 'SAI Ver.2 (64bit) Preview.2024.11.23'
    exe_hash = 'bd60d6750ef57668f9bc44eb98d992c4'
    canvas_list_offset = 0x322620

    class SAICanvasTileMap(ctypes.Structure):
        _pack_ = 1
        _fields_ = offset_fields([
            (0x08, 'tree', RPOINTER(RPOINTER(RPOINTER(ctypes.c_uint8)))),
            (0x18, 'count_x', ctypes.c_int32),
            (0x1c, 'count_y', ctypes.c_int32),
        ])

    class SAICanvas(ctypes.Structure):
        _pack_ = 1

        def get_name(self):
            return from_wide_str(self.name)

        def get_short_path(self):
            return from_wide_str(self.short_path)

    SAICanvas._fields_ = offset_fields([
            (0x000, 'next_canvas', RPOINTER(SAICanvas)),
            (0x020, 'id', ctypes.c_int32),
            (0x028, 'tile_map', RPOINTER(RPOINTER(SAICanvasTileMap))),
            (0x250, 'width', ctypes.c_int32),
            (0x254, 'height', ctypes.c_int32),
            (0x2ac, 'name', ctypes.c_uint16 * 0x100),
            (0x4ac, 'short_path', ctypes.c_uint16 * 0x100)
        ])
add_sai_api(SAIv2_API_2024_11_23)

class SAI():
    def __init__(self):
        self.proc = None
        pid = self.find_running_sai_pid()
        if pid is None:
            raise Exception('No SAI process detected.')
        self.proc = OpenProcess(pid=pid)
        self.base_address = self.get_base_address()
        self.api = get_sai_api(self.get_sai_exe_hash())(self.proc, self.base_address)

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        if self.proc is not None:
            self.proc.close()

    def get_pid(self):
        return self.proc.pid

    def find_running_sai_pid(self):
        return (util.get_process_id_by_process_name(SAIv1_API_Base.process_name)
            or  util.get_process_id_by_process_name(SAIv2_API_Base.process_name))

    def get_region_data_by_name(self, name):
        for region in self.proc.get_memory_regions():
            path = region['struct'].Path.decode()
            if name in path:
                return path, region['address']

    def get_base_address(self):
        if 'win' in sys.platform:
            return get_base_address(self.get_pid())
        else: # linux
            return self.get_region_data_by_name(psutil.Process(self.get_pid()).name())[1]

    def get_sai_exe_path(self):
        proc = psutil.Process(self.get_pid())
        if 'win' in sys.platform:
            return proc.exe()
        else: # linux
            return self.get_region_data_by_name(proc.name())[0]

    def get_sai_exe_hash(self):
        exe = self.get_sai_exe_path()
        if not exe:
            return None
        with open(exe, 'rb') as f:
            return hashlib.file_digest(f, 'md5').hexdigest()

    def is_sai_version_compatible(self, log=True):
        sai_hash = self.get_sai_exe_hash()
        compat = self.api.exe_hash == sai_hash
        if log and not compat:
            print('SAI version may not be compatible:')
            print('  Compatible versions:')
            for api in sorted(sai_api_lookup.values(), key=lambda x: x.__name__):
                print('    Version:', api.version_name)
                print('      Exe hash:  ', api.exe_hash)
            print('  Found exe hash:', sai_hash)
            print('Defaulted to version:', self.api.version_name)
        return compat

    def get_canvas_list(self):
        return self.api.get_canvas_list()

    def get_canvas_image(self, canvas):
        return self.api.get_canvas_image(canvas)

    def check_if_canvas_exists(self, canvas):
        return self.api.check_if_canvas_exists(canvas)

def test():
    with SAI() as sai:
        print('SAI compatible:', sai.is_sai_version_compatible())
        canvas = sai.get_canvas_list()[-1]
        print('Canvas name:', canvas.get_name())
        img = sai.get_canvas_image(canvas)
        import cv2
        cv2.imwrite('test.jpg', img)
