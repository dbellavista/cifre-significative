"""Figure translation must preserve numbers and reject untranslated labels."""
import matplotlib.pyplot as plt
import pytest

from cifresignificative.core.viz.translation import translate_figure


def test_dynamic_numbers_are_preserved():
    fig = plt.figure()
    text = fig.text(.1, .5, "Mediana degli altri 25: −0.90%")
    translate_figure(fig, {"Mediana degli altri 26: +3.20%": "Median of the other 26: +3.20%"})
    assert text.get_text() == "Median of the other 25: −0.90%"
    plt.close(fig)


def test_unknown_labels_fail_before_any_text_changes():
    fig = plt.figure()
    text = fig.text(.1, .5, "Italia")
    fig.text(.1, .7, "Nuova etichetta")
    with pytest.raises(ValueError, match="Missing English figure label"):
        translate_figure(fig, {"Italia": "Italy"})
    assert text.get_text() == "Italia"
    plt.close(fig)


def test_translation_cannot_change_a_number():
    fig = plt.figure()
    with pytest.raises(ValueError, match="changes numbers"):
        translate_figure(fig, {"Mediana 26": "Median 27"})
    plt.close(fig)


def test_axis_translation_survives_redraw():
    fig, ax = plt.subplots()
    ax.set_yticks([0, 1], ["Italia", "Mediana 26"])
    translate_figure(fig, {"Italia": "Italy", "Mediana 25": "Median 25"})
    fig.canvas.draw()
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Italy", "Median 26"]
    plt.close(fig)
