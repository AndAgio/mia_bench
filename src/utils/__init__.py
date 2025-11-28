from .settings import gather_settings
from .log import get_logger
from .log import DumbLogger
from .telegram import TelBotManager
from .multi_gpu import ddp_setup


def convert_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), int(s)