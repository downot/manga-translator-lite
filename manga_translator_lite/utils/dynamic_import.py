import sys
import platform
import subprocess
import importlib.util
import logging

logger = logging.getLogger('manga-translator.utils.dynamic_import')

WHEELS = {
    ('darwin', 'x86_64'): 'https://github.com/frederik-uni/manga-image-translator-rust/releases/download/v0.4.5/rusty_manga_image_translator-0.4.5-cp39-abi3-macosx_13_0_x86_64.whl',
    ('darwin', 'arm64'): 'https://github.com/frederik-uni/manga-image-translator-rust/releases/download/v0.4.5/rusty_manga_image_translator-0.4.5-cp39-abi3-macosx_14_0_arm64.whl',
    ('linux', 'aarch64'): 'https://github.com/frederik-uni/manga-image-translator-rust/releases/download/v0.4.5/rusty_manga_image_translator-0.4.5-cp39-abi3-manylinux_2_35_aarch64.whl',
    ('linux', 'x86_64'): 'https://github.com/frederik-uni/manga-image-translator-rust/releases/download/v0.4.5/rusty_manga_image_translator-0.4.5-cp39-abi3-manylinux_2_35_x86_64.whl',
    ('win32', 'AMD64'): 'https://github.com/frederik-uni/manga-image-translator-rust/releases/download/v0.4.5/rusty_manga_image_translator-0.4.5-cp39-abi3-win_amd64.whl',
}

def get_wheel_url():
    system = sys.platform
    machine = platform.machine()
    
    # Normalize machine names
    if machine.lower() in ('amd64', 'x86_64'):
        if system == 'win32':
            machine = 'AMD64'
        else:
            machine = 'x86_64'
    elif machine.lower() in ('arm64', 'aarch64'):
        if system == 'darwin':
            machine = 'arm64'
        else:
            machine = 'aarch64'
            
    return WHEELS.get((system, machine))

def install_package(url):
    logger.info(f"Installing rusty-manga-image-translator from {url}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install package: {e}")
        return False

def ensure_rusty_manga_image_translator():
    package_name = 'rusty_manga_image_translator'
    if importlib.util.find_spec(package_name) is not None:
        return importlib.import_module(package_name)
    
    url = get_wheel_url()
    if not url:
        logger.error(f"No compatible wheel found for platform {sys.platform} and machine {platform.machine()}")
        raise ImportError(f"Cannot find compatible rusty-manga-image-translator for {sys.platform} {platform.machine()}")
    
    if install_package(url):
        # Refresh import cache
        importlib.invalidate_caches()
        return importlib.import_module(package_name)
    else:
        raise ImportError("Failed to install rusty-manga-image-translator dynamically.")
