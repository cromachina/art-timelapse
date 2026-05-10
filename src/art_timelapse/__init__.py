import logging
import gettext
import importlib.resources

logging.basicConfig(
    format='[%(asctime)s][%(levelname)s] %(message)s',
    level=logging.INFO
)

lang = None

def set_locale(locale_code):
    global lang
    with importlib.resources.path('art_timelapse.locales') as path:
        lang = gettext.translation('art-timelapse', localedir=path, languages=[locale_code], fallback=locale_code == 'en')

set_locale('en')

def _(text):
    return lang.gettext(text)