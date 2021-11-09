import functools
import os
import random
import requests
import time
from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch


@contextmanager
def timer(message: str):
    print(f"[{message}] start.")
    t0 = time.time()
    yield
    elapsed_time = time.time() - t0
    print(f"[{message}] done in {elapsed_time / 60:.1f} min.")


def set_seed(seed: int = 1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def line_notification(function: Callable):
    """Decorator function for sending notification of finishing execution to LINE."""

    def send_line_notification(message: str):
        """Send notification to LINE."""
        line_token = "DCYd8FhH1uXIVsMgdtCfd95pcrXMThRkvF0qaNy9qfr"
        endpoint = "https://notify-api.line.me/api/notify"
        payload = {"message": f"\n{message}"}
        headers = {"Authorization": f"Bearer {line_token}"}
        requests.post(endpoint, data=payload, headers=headers)

    @functools.wraps(function)
    def _line_notification(*args, **kwargs):
        try:
            function(*args, **kwargs)
        except Exception as e:
            send_line_notification(e)
            raise Exception(e)
        else:
            send_line_notification("succeed.")

    return _line_notification
