from .cleaning.cleaner import Cleaner
from .cleaning.text_cleaner import TextBasedCleaner
import re

disclaimer_cleaner = Cleaner('cleaning/model/disclaimer.jl')


def custom_clean_text(text):
    text = re.sub("([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", "[EMAILID-PATTERN]", text)
    text = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', "[URL]", text)
    text = re.sub(r'\[[^ ]*\]', '', text)  # added by Vini to remove text in brackets

    txt_cleaner = TextBasedCleaner(text)
    txt_cleaner.clean_text_signoff_signature()
    txt_cleaner.clean_text_signature_by_regex()
    txt_cleaner.clean_text_salutation_name()
    txt_cleaner.clean_text_signature_by_pipe()
    txt_cleaner.clean_text_forward_by_regex()
    txt_cleaner.clean_text_code_constructs()
    text = txt_cleaner.get_text()
    text = re.sub('\d+', '<NUM>', text)
    text = text.replace('\n', ' ')  # added by Vini
    text = text.replace('[ ]+', ' ')  # added by Vini

    return text.strip()

def clean_description(description):
    lines = [string.strip() for string in re.split('\x01|\xa0',description) if len(string.strip())>1]
    text = "\n".join(lines).strip()

    text = disclaimer_cleaner.clean_disclaimer(text)
    text = custom_clean_text(text)
    return text


subject_pats = ['re:', 'fw:', 'fwd:', 'urgent']


def clean_subject(subject):
    lines = [string.strip() for string in re.split('\x01|\xa0', subject) if len(string.strip()) > 1]
    text = "\n".join(lines).strip()

    text_lower = text.lower()
    for pat in subject_pats:
        if pat in text_lower:
            index = text_lower.index(pat)
            text = text[:index] + text[index + len(pat):]

    text = custom_clean_text(text)

    return text