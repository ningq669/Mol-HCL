import pickle as pkl
import torch
import numpy as np

def load_pkl_as_tensor(file_path: str, device: str = "cpu") -> torch.FloatTensor:
    """
    读取 pkl 文件并转换为 PyTorch Tensor，以兼容神经网络模型输入（不修改特征维度）。

    :param file_path: pkl 文件路径
    :param device: 设备 ("cpu" 或 "cuda")
    :return: PyTorch Tensor，形状为 (num_samples, num_features)
    """
    try:
        # 检查设备是否有效
        if device not in ["cpu", "cuda"]:
            print(f"⚠️ 警告: 无效设备类型 '{device}'，已自动切换为 'cpu'！")
            device = "cpu"

        # 读取 pkl 文件（使用 'latin1' 编码以兼容不同版本）
        with open(file_path, 'rb') as f:
            data = pkl.load(f, encoding="latin1")

        # 如果数据是稀疏矩阵，则转换为 NumPy 数组
        if hasattr(data, "toarray"):
            data = data.toarray()

        # 确保数据为 NumPy 数组，并转换为 float32（适用于 PyTorch）
        data = np.array(data, dtype=np.float32)

        # 转换为 PyTorch Tensor，并移动到指定设备
        tensor_data = torch.tensor(data).to(device)

        return tensor_data

    except Exception as e:
        print(f"❌ 读取 {file_path} 时出错: {e}")
        return None
