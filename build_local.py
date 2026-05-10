from setuptools import Distribution
from setuptools_gettext import build_mo, load_pyproject_config

dist = Distribution()
load_pyproject_config(dist, { "build_dir": "src/art_timelapse/locales" })
cmd = build_mo(dist)
cmd.initialize_options()
cmd.finalize_options()
cmd.output_base = 'art-timelapse'
cmd.run()
