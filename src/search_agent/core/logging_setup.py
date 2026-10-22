"""统一 root logger，仅写 UTF-8 文件；CLI 终端交互仍使用自身的 print/Rich。"""

import logging

from .config import LoggingConfig, get_config

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logger(config: LoggingConfig | None = None) -> None:
    """设置全局文件日志，同一路径重复调用不增加 handler 或重复记录。

    各模块用 logging.getLogger(__name__)，通过 root 统一格式与级别。
    替换 root 上旧的 handler，保证无控制台输出。不记录密钥或整个 AppConfig。
    """
    config = config or get_config().logging
    path = config.path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(config.level)
    if (
        len(root.handlers) == 1
        and isinstance(root.handlers[0], logging.FileHandler)
        and root.handlers[0].baseFilename == str(path)
    ):
        handler = root.handlers[0]
    else:
        for existing in root.handlers[:]:
            root.removeHandler(existing)
            existing.close()
        handler = logging.FileHandler(path, encoding="utf-8")
        root.addHandler(handler)
    handler.setLevel(config.level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
