# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Framework adapters for Holographic Memory. Each lives in its own subpackage and imports its framework lazily."""

from .core import *  # noqa: F401,F403
from .core import __all__

__version__ = "0.1.0"
