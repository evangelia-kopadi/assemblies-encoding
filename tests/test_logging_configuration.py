import io
import logging

from src.logging_configuration import LOGGER_NAMESPACE, configure_logging, get_logger


def _flush_repo_logger():
    for handler in logging.getLogger(LOGGER_NAMESPACE).handlers:
        handler.flush()


def test_configure_logging_keeps_readable_message_only_output():
    stream = io.StringIO()
    configure_logging(stream=stream, force=True)

    get_logger("src.example").info("hello experiment")

    assert stream.getvalue() == "hello experiment\n"


def test_configure_logging_supports_quiet_mode_and_file_logs(tmp_path):
    stream = io.StringIO()
    log_file = tmp_path / "sweep.log"
    configure_logging(level="WARNING", stream=stream, log_file=log_file, force=True)

    logger = get_logger("src.example")
    logger.info("hidden progress")
    logger.warning("visible warning")
    _flush_repo_logger()

    assert "hidden progress" not in stream.getvalue()
    assert "visible warning" in stream.getvalue()

    file_text = log_file.read_text(encoding="utf-8")
    assert "hidden progress" not in file_text
    assert "visible warning" in file_text
