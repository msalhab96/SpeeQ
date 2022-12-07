import json


def load_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data


def load_text(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        data = f.read()
    return data
