import os
import datetime as dt
import logging
from typing import (
    Any,
    Optional,
    Tuple,
)

import yaml


logger: logging.Logger = logging.getLogger(__name__)

APP_TITLE_PREFIX = "AI Agent"

USER_AVATAR_PATH = None
AGENT_AVATAR_PATH = None

_translations: Optional[dict] = None


# TODO - avoid concurrent access to the database

_SUPPORTED_LANGUAGES: Tuple[str] = ()
_DEFAULT_LANGUAGE = "en"
_selected_language: Optional[str] = None

def set_language(language: str):
    global _translations
    if _translations is None:
        load_translations()
    if language in _SUPPORTED_LANGUAGES:
        global _selected_language
        _selected_language = language
    else:
        _selected_language = _DEFAULT_LANGUAGE


def load_translations():
    global _translations
    logger.debug(f"Loading translations for {_selected_language}")
    try:
        with open("translations.yaml", "r", encoding="utf-8") as f:
            _translations = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading translations: {str(e)}")
        _translations = {}


def get_text(key: str) -> str:
    global _translations
    # TODO: get language from session state
    global _selected_language
    try:
        return _translations[_selected_language][key]
    except KeyError:
        logger.error(
            f"Missing translation for key: {key} in language: {_selected_language}")
        return key

