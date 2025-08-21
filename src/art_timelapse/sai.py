import ctypes
import hashlib
import sys

import psutil
from PyMemoryEditor import OpenProcess
import numpy as np

def from_wide_str(data):
    return bytes(data).decode('utf-16le')

class RemotePointer(ctypes._Pointer):
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        unpack = lambda p, i=0, s=1:(p, i, s)
        proc, index, count = unpack(*key)
        dtype = self._type_
        dsize = ctypes.sizeof(dtype)
        offset = dsize * index
        byte_size = dsize * count
        address = self.get_address() + offset
        data = proc.read_process_memory(address, bytes, byte_size)
        if count == 1:
            result = self._type_.from_buffer_copy(data)
        else:
            result = (self._type_ * count).from_buffer_copy(data)
        if isinstance(result, ctypes._SimpleCData):
            return result.value
        return result

    def get_address(self):
        return ctypes.cast(self, ctypes.c_void_p).value

_remote_pointer_cache = {}
def RPOINTER(dtype):
    if dtype in _remote_pointer_cache:
        return _remote_pointer_cache[dtype]
    name = f'RP_{ dtype.__qualname__ }'
    ptype = type(name, (RemotePointer,), { '_type_': dtype })
    _remote_pointer_cache[dtype] = ptype
    return ptype

target_process_name = 'sai2.exe'

sai_api_lookup = {}
def add_api_lookup(api):
    sai_api_lookup[api.exe_hash] = api

def get_sai_api(hash):
    result = sai_api_lookup.get(hash)
    if result is None:
        result = sorted(sai_api_lookup.values(), key=lambda x: x.__name__)[-1]
    return result

# Constant for all Windows programs.
base_address = 0x1_4000_0000

pad_index = 0
def pad(size):
    global pad_index
    field = (f'__pad{pad_index}', ctypes.c_char * size)
    pad_index += 1
    return field

# Make a new version of this class for each new SAI version.
class SAI_API_2024_11_23:
    version_name = 'SAI Ver.2 (64bit) Preview.2024.11.23'
    exe_hash = 'bd60d6750ef57668f9bc44eb98d992c4'
    canvas_list_address = base_address + 0x322620

    class SAICanvasTileMap(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            pad(0x8),
            ('tree', RPOINTER(RPOINTER(RPOINTER(ctypes.c_uint8)))),
            pad(0x8),
            ('count_x', ctypes.c_int32),
            ('count_y', ctypes.c_int32),
        ]

    class SAICanvas(ctypes.Structure):
        _pack_ = 1

        def get_name(self):
            return from_wide_str(self.name)

    SAICanvas._fields_ = [
            ('next_canvas', RPOINTER(SAICanvas)),
            pad(0x18),
            ('id', ctypes.c_int32),
            pad(0x4),
            ('tile_map', RPOINTER(RPOINTER(SAICanvasTileMap))), # 0x28
            pad(0x220),
            ('width', ctypes.c_int32), # 0x250
            ('height', ctypes.c_int32), # 0x254
            pad(0x54),
            ('name', ctypes.c_uint8 * 0x200) # 0x2ac
        ]
add_api_lookup(SAI_API_2024_11_23)

class SAI():
    def __init__(self, proc=None):
        self._proc = proc
        if self._proc is None:
            self._proc = OpenProcess(process_name = target_process_name)
        self.api = get_sai_api(self.get_sai_exe_hash())

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        if self._proc is not None:
            self._proc.close()

    def get_pid(self):
        return self._proc.pid

    def get_sai_exe_path(self):
        if 'win' in sys.platform:
            return psutil.Process(self.get_pid()).exe()
        else:
            for region in self._proc.get_memory_regions():
                if region['address'] == base_address:
                    return region['struct'].Path.decode()

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
        proc = self._proc
        result = []
        canvas_ptr = ctypes.cast(self.api.canvas_list_address, RPOINTER(RPOINTER(self.api.SAICanvas)))[proc]
        while canvas_ptr:
            canvas = canvas_ptr[proc]
            result.append(canvas)
            canvas_ptr = canvas.next_canvas
        return result

    def get_canvas_image(self, canvas):
        proc = self._proc
        tile_map = canvas.tile_map[proc][proc]
        height = canvas.height
        width = canvas.width
        tiles_y = tile_map.count_y
        tiles_x = tile_map.count_x
        image = np.empty(shape=(tiles_y * 256, tiles_x * 256, 4), dtype=np.uint8)
        tile_shape = (256, 256, 4)
        tile_size = 256 * 256 * 4
        for y in range(tiles_y):
            for x in range(tiles_x):
                tile = tile_map.tree[proc, y][proc, x][proc, 0, tile_size]
                tile = np.frombuffer(tile, dtype=np.uint8).reshape(tile_shape)
                image[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256] = tile
        return image[0:height,0:width,:3]

    def check_if_canvas_exists(self, canvas):
        return any((c.id == canvas.id for c in self.get_canvas_list()))

def test():
    with SAI() as sai:
        print('SAI compatible:', sai.is_sai_version_compatible())
        canvas = sai.get_canvas_list()[1]
        print('Canvas name:', canvas.get_name())
        img = sai.get_canvas_image(canvas)
        print(img.shape)