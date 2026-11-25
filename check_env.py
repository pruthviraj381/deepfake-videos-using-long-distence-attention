import sys

packages = ['flask', 'torch', 'cv2', 'numpy', 'werkzeug', 'requests', 'PIL']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'installed')
        print(f"{pkg}: {ver}")
    except ImportError as e:
        print(f"{pkg}: NOT INSTALLED ({e})")
