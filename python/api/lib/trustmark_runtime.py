from typing import Any

from python.api.lib.schemas import DeviceEnum, ModelTypeEnum

try:
    from trustmark import TrustMark
except ImportError:  # Local repo fallback when package is not installed.
    from python.trustmark import TrustMark


def runtime_device(device: DeviceEnum) -> str:
    if device == DeviceEnum.CPU:
        return "cpu"
    return "cuda:0"


def build_trustmark(model_type: ModelTypeEnum, device: DeviceEnum) -> Any:
    return TrustMark(
        verbose=False,
        model_type=model_type.value,
        device=runtime_device(device),
    )
