from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_PATH / 'data'
ZIP_PATH = DATA_DIR / "MOT17.zip"
CACHE_DIR = DATA_DIR / 'mot_cache'