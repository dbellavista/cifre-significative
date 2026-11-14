"""Translate rendered figure labels without changing their numerical content."""
import re

from matplotlib.figure import Figure
from matplotlib.text import Text

NUMBER = re.compile(r"[+−-]?\d+(?:[.,]\d+)?")


def _template(text: str) -> tuple[str, list[str]]:
    values = NUMBER.findall(text)
    parts = NUMBER.split(text)
    return "".join(part + ("{" + str(i) + "}" if i < len(values) else "") for i, part in enumerate(parts)), values


def translate_figure(fig: Figure, labels: dict[str, str]) -> None:
    """Replace all text using numeric templates; reject missing labels or changed numbers.

    The caller owns language selection, layout and saving. Unknown alphabetic labels
    fail before a file is saved, so new source labels cannot silently remain Italian.
    """
    templates = {}
    for source, target in labels.items():
        key, numbers = _template(source)
        value, translated_numbers = _template(target)
        if numbers != translated_numbers:
            raise ValueError(f"Translation changes numbers: {source!r} -> {target!r}")
        if key in templates and templates[key] != value:
            raise ValueError(f"Conflicting translation: {source!r}")
        templates[key] = value
    fig.canvas.draw()
    replacements = []
    for artist in fig.findobj(Text):
        source = artist.get_text()
        if not any(c.isalpha() for c in source):
            continue
        key, numbers = _template(source)
        if key not in templates:
            raise ValueError(f"Missing English figure label: {source!r}")
        replacements.append((artist, templates[key].format(*numbers)))
    for artist, target in replacements:
        artist.set_text(target)
    # Axis formatters recreate tick strings on every draw; update their source too.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            texts = [tick.get_text() for tick in axis.get_ticklabels()]
            if any(any(c.isalpha() for c in text) for text in texts):
                translated = []
                for text in texts:
                    key, numbers = _template(text)
                    translated.append(templates[key].format(*numbers) if any(c.isalpha() for c in text) else text)
                axis.set_ticks(axis.get_ticklocs(), labels=translated)
