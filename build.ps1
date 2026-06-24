# build.ps1
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist, build, src\PyFAME.egg-info
python -m build