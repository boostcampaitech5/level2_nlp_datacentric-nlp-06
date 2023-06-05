import re
import pandas as pd
from tqdm import tqdm

from hanspell import spell_checker


_illegal_xml_chars_RE = re.compile(u'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]')
XML_PREDEFINED_ENTITIES = {
	"<": "&#60;",
	"&": "&#38;",
	">": "&#62;",
	"'": "&#39;",
	'"': "&#34;",
}

def escape_xml_illegal_chars(val, replacement='?'):
    """Filter out characters that are illegal in XML.

    Looks for any character in val that is not allowed in XML
    and replaces it with replacement ('?' by default).

    >>> escape_illegal_chars("foo \x0c bar")
    'foo ? bar'
    >>> escape_illegal_chars("foo \x0c\x0c bar")
    'foo ?? bar'
    >>> escape_illegal_chars("foo \x1b bar")
    'foo ? bar'
    >>> escape_illegal_chars(u"foo \uFFFF bar")
    u'foo ? bar'
    >>> escape_illegal_chars(u"foo \uFFFE bar")
    u'foo ? bar'
    >>> escape_illegal_chars(u"foo bar")
    u'foo bar'
    >>> escape_illegal_chars(u"foo bar", "")
    u'foo bar'
    >>> escape_illegal_chars(u"foo \uFFFE bar", "BLAH")
    u'foo BLAH bar'
    >>> escape_illegal_chars(u"foo \uFFFE bar", " ")
    u'foo   bar'
    >>> escape_illegal_chars(u"foo \uFFFE bar", "\x0c")
    u'foo \x0c bar'
    >>> escape_illegal_chars(u"foo \uFFFE bar", replacement=" ")
    u'foo   bar'
    """
    return _illegal_xml_chars_RE.sub(replacement, val)


def clean_spell(data):
    """
    hanspell을 활용하여 맞춤법 정제를 해주는 함수

    inputs:
        원본 데이터

    returns:
        맞춤법 정제가 완료된 text들로 대체된 데이터
    """
    raw_text = data['text']
    checked_text = []

    for text in tqdm(raw_text):
        text =  escape_xml_illegal_chars(text)
        for char, escape_char in XML_PREDEFINED_ENTITIES.items():
            text = text.replace(char, escape_char)
        checked_text.append(spell_checker.check(text)[2])

    data['text'] = checked_text

    return data
