"""
デバイス設定のユーティリティモジュール
"""

import torch
from typing import Optional

def get_device() -> torch.device:
    """
    利用可能な最適なデバイスを取得
    
    Returns:
        torch.device: 使用するデバイス
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def move_to_device(model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    モデルを指定されたデバイスに移動
    
    Args:
        model: 移動対象のモデル
        device: 移動先のデバイス（Noneの場合は自動選択）
        
    Returns:
        デバイスに移動されたモデル
    """
    if device is None:
        device = get_device()
    
    return model.to(device) 