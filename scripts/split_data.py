import dpdata
import numpy as np
import shutil
import os

data_system = dpdata.LabeledSystem("training_data", fmt="deepmd/npy")
total = len(data_system)
print(f"总帧数: {total}")

indices = np.random.permutation(total)
n_train = int(0.8 * total)
n_val = int(0.1 * total)

for d in ["train_set", "val_set", "test_set"]:
    if os.path.exists(d):
        shutil.rmtree(d)

train_sys = data_system.sub_system(indices[:n_train])
val_sys = data_system.sub_system(indices[n_train:n_train+n_val])
test_sys = data_system.sub_system(indices[n_train+n_val:])

train_sys.to_deepmd_npy("train_set")
val_sys.to_deepmd_npy("val_set")
test_sys.to_deepmd_npy("test_set")

print(f"训练集: {len(train_sys)} 帧")
print(f"验证集: {len(val_sys)} 帧")
print(f"测试集: {len(test_sys)} 帧")
