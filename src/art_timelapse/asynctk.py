import asyncio
import time
import contextlib
import ttkbootstrap as ttk

@contextlib.asynccontextmanager
async def sync_interval(interval):
    start = time.perf_counter()
    try:
        yield
    except Exception as ex:
        raise ex
    else:
        delta = time.perf_counter() - start
        sleep_time = interval - delta
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

class AsyncTk(ttk.App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol('WM_DELETE_WINDOW', self.stop)
        self.running = False
        self.update_interval = 1.0 / 60

    def cleanup(self):
        pass

    def stop(self):
        self.running = False
        self.cleanup()

    async def async_main_loop(self):
        self.running = True
        while self.running:
            async with sync_interval(self.update_interval):
                self.update()
                await asyncio.sleep(0)

class AsyncTkCallback:
    tasks = set()

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        task = asyncio.create_task(self.func(*args, **kwargs))
        AsyncTkCallback.tasks.add(task)
        task.add_done_callback(AsyncTkCallback.tasks.discard)