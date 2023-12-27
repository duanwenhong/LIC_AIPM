import json
import torch
from pathlib import Path


class Checkpoints(object):
    """
    checkpoints managers
    """
    def __init__(self, length, root):
        super(Checkpoints, self).__init__()
        self._length = int(length)
        self._file_list = list()
        self._root = Path(root)
        self._record = self._root / "checkpoints.json"

        if self._record.exists():
            with open(self._record, 'r') as f:
                for item in json.load(f):
                    file = self._root / Path(item)
                    if file.exists():
                        self._file_list.append(file)
            if len(self._file_list) == 0:
                raise ValueError("Path in checkpoint do not exist!")

    @property
    def length(self):
        return self._length
    # duan modify the save pth file --不进行删除 动态保存所有结果
    def save(self, file, data):
        torch.save(data, file)
        self._file_list.insert(0, file)
        if len(self._file_list) > self.length:
            del_file = self._file_list.pop()
            del_file.unlink()
        with open(self._record, 'w') as f:
            json.dump([item.name for item in self._file_list], f)

    @property
    def file(self):
        return self._file_list[0]

    def is_exists(self):
        if len(self._file_list) == 0:
            return False
        else:
            return self.file.exists()
