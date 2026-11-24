"""Use the audited GPU guard and recheck transient handoffs."""

from datetime import datetime
import importlib.util
import time
from zoneinfo import ZoneInfo

from common import OUTPUTS

spec = importlib.util.spec_from_file_location("prior_gpu_driver", OUTPUTS/"continual-bench-next/driver.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def cap():
    return 6 if datetime.now(ZoneInfo("Asia/Shanghai")).weekday() >= 5 else 4


driver.launch_cap = cap


def execute(args, gpus=1, timeout=14400):
    deadline = time.monotonic()+35
    while True:
        try:
            return driver.execute(args,gpus,timeout)
        except RuntimeError as e:
            if not str(e).startswith("Unsafe GPU handoff:") or time.monotonic()>=deadline:
                raise
            print("Waiting for GPU handoff to settle.",flush=True)
            time.sleep(4)
