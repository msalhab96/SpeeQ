import os
import platform
from typing import List
from constants import FileKeys


def clear():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def get_text_list(data: List[dict]) -> List[str]:
    return [item[FileKeys.text_key] for item in data]
