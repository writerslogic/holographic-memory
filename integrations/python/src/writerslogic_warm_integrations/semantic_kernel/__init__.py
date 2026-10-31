# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Semantic Kernel vector store. Requires `semantic-kernel`."""

from .collection import WarmCollection, WarmStore

__all__ = ["WarmCollection", "WarmStore"]

HolographicCollection = WarmCollection
HolographicStore = WarmStore
__all__ += ['HolographicCollection', 'HolographicStore']
