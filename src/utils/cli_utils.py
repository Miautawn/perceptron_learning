from typing import List, Callable

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

def cli_select(message: str, choices: List):
    return_value = inquirer.select(
        message=message,
        choices=choices,
    ).execute()

    return return_value

def cli_text(message: str, validation_function: Callable, transform_function: Callable):
    return_value = inquirer.text(
        message=message,
        validate=validation_function,
        filter=transform_function
    ).execute()

    return return_value