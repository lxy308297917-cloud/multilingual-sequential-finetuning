"""Allow the driver utilization window to settle between child processes."""

import time

from driver import execute


def execute_checked(args, gpus=1, timeout=10800):
    deadline=time.monotonic()+30
    while True:
        try:
            return execute(args,gpus,timeout)
        except RuntimeError as error:
            if not str(error).startswith('Unsafe GPU handoff:') or time.monotonic()>=deadline:
                raise
            print('GPU handoff pending; will recheck all process and memory conditions.',flush=True)
            time.sleep(4)
