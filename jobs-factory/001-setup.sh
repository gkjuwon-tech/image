# factory box setup: gsplat + trainer deps + COLMAP (CPU) for triangulating DL3DV scenes
set -e
. jobs/lib.inc
apt-get update -qq && apt-get install -y -qq git curl unzip aria2 libgl1 libglib2.0-0 ninja-build colmap sqlite3 >/dev/null
python -V; python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"
cd $W && [ -d gsplat ] || git clone -q --recursive -b v1.5.3 https://github.com/nerfstudio-project/gsplat.git
cd gsplat && MAX_JOBS=24 pip install -q --no-build-isolation . 2>&1 | tail -3
pip install -q -r examples/requirements.txt 2>&1 | tail -3
python -c "import gsplat, fused_ssim; print('gsplat', gsplat.__version__)"
