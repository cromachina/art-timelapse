import asyncio
import tkinter as tk

class AsyncTk(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol('WM_DELETE_WINDOW', self.stop)
        self.running = False
        self.sleep_time = 0.01

    def cleanup(self):
        pass

    def stop(self):
        self.running = False
        self.cleanup()

    async def async_main_loop(self):
        self.running = True
        while self.running:
            self.update()
            await asyncio.sleep(self.sleep_time)

class AsyncTkCallback:
    tasks = set()

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        task = asyncio.create_task(self.func(*args, **kwargs))
        AsyncTkCallback.tasks.add(task)
        task.add_done_callback(AsyncTkCallback.tasks.discard)