import json
from typing import Union


def load_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data


def load_text(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = f.read()
    return data


def save_json(
        file_path,
        data: Union[dict, list],
        encoding='utf-8'
        ) -> None:
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f)


def save_text(
        file_path,
        data: str,
        encoding='utf-8'
        ) -> None:
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(data)
