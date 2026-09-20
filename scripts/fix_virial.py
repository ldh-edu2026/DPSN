import numpy as np
import os

# ---------- 读取 .stress 文件（单位：bar）----------
def read_stress_file(stress_file):
    stresses = []
    with open(stress_file) as f:
        lines = f.readlines()
    for line in lines:
        if not line.strip() or line.startswith('|') or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 11 and parts[0].isdigit():
            # 列：Step Time xx xy xz yx yy yz zx zy zz
            try:
                stress_components = [float(x) for x in parts[2:11]]
                stresses.append(stress_components)
            except:
                pass
    return np.array(stresses)

# ---------- 读取 .cell 文件（盒子）----------
def read_cell_file(cell_file):
    boxes = []
    with open(cell_file) as f:
        lines = f.readlines()
    for line in lines[1:]:   # 跳过表头
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 11:
            # 格式: Step Time Ax Ay Az Bx By Bz Cx Cy Cz
            boxes.append([float(x) for x in parts[2:11]])
    return np.array(boxes)

# ---------- 读取现有 virial.npy（错误单位，bar）----------
old_virial = np.load('training_data/set.000/virial.npy')

# ---------- 重新从原始 .stress 读取，并计算正确单位 ----------
stress_file = "Si3N4-hp-md-1.stress"
cell_file = "Si3N4-hp-md-1.cell"

stress_bar = read_stress_file(stress_file)
boxes = read_cell_file(cell_file)

# 确保帧数一致
nframes = min(stress_bar.shape[0], boxes.shape[0])
stress_bar = stress_bar[:nframes]
boxes = boxes[:nframes]

# 计算体积：将盒子 9 分量转为 3x3 矩阵，取行列式
volumes = np.linalg.det(boxes.reshape(-1, 3, 3))  # (nframes,)

# 转换为 virial (eV)
# 1 bar = 1e-6 eV/Å³，virial = stress (bar) * 1e-6 * volume (Å³)
virial_eV = stress_bar * 1e-6 * volumes[:, None]  # (nframes, 9)

print(f"原 virial 第一帧: {old_virial[0]}")
print(f"新 virial 第一帧: {virial_eV[0]}")
print(f"体积范围: {volumes.min():.2f} ~ {volumes.max():.2f} Å³")

# 备份旧 virial.npy
os.rename('training_data/set.000/virial.npy', 'training_data/set.000/virial_old_backup.npy')
# 保存新 virial
np.save('training_data/set.000/virial.npy', virial_eV)
print("✅ 修正完成！新的 virial.npy 已保存。")
