import json

import pytest

pytest.importorskip("ipywidgets")

from propaxis import Component, Construct
from propaxis.notebook import construct_editor


def test_editor_roundtrip_preserves_every_setting():
    spec = Construct(
        "test",
        [
            Component(
                "first", ["one", "two"], ["three"], hypotheses=["claim"], description="meaning"
            )
        ],
        window=3,
        top_k=None,
        temperature=0.3,
        max_words=45,
        combine="minimum",
    )
    editor = construct_editor(spec)
    assert editor.build() == spec
    editor._preview()
    assert Construct.from_dict(json.loads(editor.preview.value)) == spec
    editor.fields[0]["positive"].value += "\nfour"
    assert len(editor.build().components[0].positive) == 3
    assert len(spec.components[0].positive) == 2


def test_editor_validation_add_remove():
    editor = construct_editor()
    editor._preview()
    assert "Check the specification" in editor.status.value
    editor.add_component(Component("second", ["yes"], ["no"]))
    assert len(editor.fields) == 2
    editor.tabs.selected_index = 0
    editor.remove_component()
    assert editor.build().components[0].name == "second"


def test_editor_escapes_validation_errors():
    editor = construct_editor()
    editor.fields[0]["name"].value = "<script>alert(1)</script>"
    editor._preview()
    assert "<script>" not in editor.status.value
