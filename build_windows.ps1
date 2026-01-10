winget install python
rm build -r -Force
rm dist -r -Force
python -m venv venv
. venv/Scripts/activate.ps1
pip install -e .[dev]
pyinstaller art_timelapse.spec
$VERSION = python -c "from importlib.metadata import version; print(version('art-timelapse'), end='')"
tar -c -a -f art-timelapse-windows-v$VERSION.zip -C dist art-timelapse