from .tools import get_warm_tool, WarmDependencies
__all__ = ["get_warm_tool", "WarmDependencies"]

HolographicDependencies = WarmDependencies
__all__ += ['HolographicDependencies']

get_holographic_tool = get_warm_tool
