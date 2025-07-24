try:
    from ultrack_td.__about__ import __version__
except ImportError:
    # Fallback for development installs without proper build
    __version__ = "unknown"
