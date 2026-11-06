"""K3 Abliterator: find and remove refusal directions in Kimi K3, through colibri."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("k3-abliterator")
except PackageNotFoundError:   # running from a source checkout
    __version__ = "0.0.0"
APP_ID = "io.github.ssjrocks.K3Abliterator"
APP_NAME = "K3 Abliterator"
REPO = "ssjrocks/k3-abliterator"
