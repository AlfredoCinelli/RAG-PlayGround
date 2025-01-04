"""
Module containig utility functions.

The main functions defined in the module are:

* check_string_list: Checks if all elements in a list are strings.
* clean_text: Cleans text by removing non-alphanumeric characters and converting to lowercase.
* apply_custom_css: Applies custom CSS styling to the page and sidebar.
"""

# Import packages and modules

import re
from typing import Any, List, Literal

from src.logging import logger

# Define UDFs


def check_string_list(
    elements: List[Any],
) -> bool:
    """
    Function to check if all elements in a list are strings.

    :param elements: list of elements to check
    :type elements: List[Any]
    :return: True if all elements are strings, False otherwise
    :rtype: bool
    """
    return all(isinstance(element, str) for element in elements)


def clean_text(text: str, print_log: bool = True) -> str:
    """
    Function to clean OCR-extracted text by removing unnecessary newlines,
    hyphens, and correcting common OCR errors.

    :param text: the text to clean
    :type text: str
    :param print_log: whether to print a log message, defaults to True
    :type print_log: bool, optional
    :return: the cleaned text
    :rtype: str
    """
    # Remove hyphens at line breaks (e.g., 'exam-\nple' -> 'example')
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Replace newlines within sentences with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n+", "\n", text)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_text = text.strip()
    if print_log is True:
        logger.info("Text cleaned.")
    return cleaned_text


def apply_custom_css(
    page: Literal["main_page", "chat_page", "document_page"],
) -> str:
    """
    Function to apply custom CSS format to different pages of the application.

    :param page: which page to apply the CSS to, based on the page a different
    CSS is applied
    :type page: Literal["main_page", "chat_page", "document_page"]
    :return: string representing the CSS to apply to the page
    :rtype: str
    :raises ValueError: if the page parameter is not one of the expected values
    """
    match page:
        case "main_page":
            css_format = """
            <style>
            /* Main background and text colors */
            body {
                background-color: #f0f8ff;  /* Light cyan background */
                color: #000000;  /* Black text */
            }
            .sidebar .sidebar-content {
                background-color: #006d77;
                color: white;
                padding: 20px;
                border-right: 2px solid #003d5c;
            }
            .sidebar h2, .sidebar h4 {
                color: white;
            }
            .block-container {
                background-color: #f8f9fa;  /* Light gray background */
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            }
            .stColumn {
                text-align: center;
            }
            .footer-text {
                font-size: 1.1rem;
                font-weight: bold;
                color: #000000;
                text-align: center;
                margin-top: 10px;
            }
            .stButton button {
                background-color: #118ab2;
                color: white;
                border-radius: 5px;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
            }
            .stButton button:hover {
                background-color: #07a6c2;
                color: white;
            }
            /* Headings inside the main page */
            h1, h2, h3, h4 {
                color: #118ab2;  /* Light Blue headings */
            }
            /* Additional text styling for better visibility */
            p, div {
                color: #000000;  /* Ensure all text is black */
            }
            </style>
            """
        case "chat_page":
            css_format = """
            <style>
            /* Main background and text colors */
            body { background-color: #f0f8ff; color: #000000; }
            .sidebar .sidebar-content {
                background-color: #006d77;
                color: white;
                padding: 20px;
                border-right: 2px solid #003d5c;
            }
            .sidebar h2, .sidebar h4 { color: white; }
            .block-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            }
            .footer-text {
                font-size: 1.1rem;
                font-weight: bold;
                color: black;
                text-align: center;
                margin-top: 10px;
            }
            .stButton button {
                background-color: #118ab2;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
            }
            .stButton button:hover { background-color: #07a6c2; color: white; }
            h1, h2, h3, h4 { color: #118ab2; }
            .stChatMessage {
                background-color: grey;
                color: #000000;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .stChatMessage.user { background-color: #000000; color: #000000; }
            p {
                color: white;
            }
            div {
                color: white;
            }
            </style>
            """
        case "document_page":
            css_format = """
            <style>
            /* Main background and text colors */
            body {
                background-color: #f0f8ff;
                color: white;
            }

            .sidebar .sidebar-content {
                background-color: #006d77;
                color: #87CEEB;
                padding: 20px;
                border-right: 2px solid #003d5c;
            }

            .sidebar h2, .sidebar h4 {
                color: white;
            }

            .block-container {
                background-color: #f8f9fa;
                color: black;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            }

            .stColumn {
                text-align: center;
            }

            .footer-text {
                font-size: 1.1rem;
                font-weight: bold;
                color: white;
                text-align: center;
                margin-top: 10px;
            }

            .stButton button {
                background-color: #f8f9fa;
                color: white;
                border-radius: 5px;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
            }

            .stButton button:hover {
                background-color: #e9d2d2;
                color: white;
                border-radius: 5px;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
            }

            /* Button label text color */
            .stButton + div {
                color: white;
            }

            h1, h2, h3, h4 {
                color: #118ab2;
            }

            p {
                color: grey;
            }
            div {
                color: grey;
            }
            </style>
            """
        case _:
            raise ValueError(f"Invalid page type, {page} is not supported.")

    return css_format
