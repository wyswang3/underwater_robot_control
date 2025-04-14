# utils/log_helper.py
import logging, importlib

def setup_logging():
    from config import Config
    cfg = Config()
    level = logging.DEBUG if cfg.DEBUG else logging.INFO

    root = logging.getLogger()
    if root.handlers:                      # 若已有 handler，先全部移除
        for h in root.handlers[:]:
            root.removeHandler(h)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
