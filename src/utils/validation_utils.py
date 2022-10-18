from typing import Any

def is_float(number: Any) -> bool:
    try:
        float(number)
        return True
    except:
        return False

def is_integer(number: Any) -> bool:
    try:
        int(number)
        return True
    except:
        return False