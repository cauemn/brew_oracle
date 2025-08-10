import importlib
import logging


def test_settings_log_level_from_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    from brew_oracle.utils.config import Settings
    s = Settings()
    assert s.LOG_LEVEL == logging.DEBUG


def test_modules_share_logger(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    import brew_oracle.knowledge.beerxml_kb as beerxml_kb
    import brew_oracle.knowledge.pdf_kb as pdf_kb
    importlib.reload(pdf_kb)
    importlib.reload(beerxml_kb)
    assert pdf_kb.logger is beerxml_kb.logger
    assert pdf_kb.logger.level == logging.WARNING
