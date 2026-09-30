"""Optional notebook editor. Importing propaxis never imports notebook dependencies."""

from __future__ import annotations

import html
import json

from .spec import Component, Construct


class ConstructEditor:
    """Edit propositions in Jupyter, validate, preview JSON, then build a Construct.

    No models load and no network calls occur in the editor. Call ``build()`` to
    read current fields; it raises normal validation errors if input is invalid.
    """

    def __init__(self, construct=None):
        try:
            import ipywidgets as w
        except ImportError as error:
            raise ImportError(
                "Install the notebook extra: pip install 'propaxis[notebook]'"
            ) from error
        self._w = w
        self.name = w.Text(
            value=construct.name if construct else "my_construct", description="Name"
        )
        self.window = w.BoundedIntText(
            value=construct.window if construct else 1, min=1, max=100, description="Sentences"
        )
        self.top_k = w.BoundedIntText(
            value=(construct.top_k or 1) if construct else 1,
            min=1,
            max=1000,
            description="Top windows",
        )
        self.use_all = w.Checkbox(
            value=construct is not None and construct.top_k is None,
            description="Average all windows",
        )
        self.combine = w.Dropdown(
            options=["geometric_mean", "mean", "minimum"],
            value=construct.combine if construct else "geometric_mean",
            description="Combine",
        )
        self.max_words = w.BoundedIntText(
            value=construct.max_words if construct else 60, min=1, max=10000, description="Word cap"
        )
        self.temperature = w.BoundedFloatText(
            value=construct.temperature if construct else 0.25,
            min=0.001,
            max=100,
            description="Temperature",
        )
        self.fields = []
        self.tabs = w.Tab()
        self.status = w.HTML()
        self.preview = w.Textarea(
            description="JSON", layout=w.Layout(width="100%", height="200px"), disabled=True
        )
        add = w.Button(description="Add component", icon="plus")
        remove = w.Button(description="Remove selected", icon="minus")
        validate = w.Button(
            description="Validate and preview", button_style="primary", icon="check"
        )
        add.on_click(lambda _: self.add_component())
        remove.on_click(lambda _: self.remove_component())
        validate.on_click(lambda _: self._preview())
        self.widget = w.VBox(
            [
                w.HTML(
                    "<h3>Build a text measure</h3><p>Write one complete proposition per line. Positive and contrast poles should express competing claims about the same subject.</p>"
                ),
                self.name,
                self.tabs,
                w.HBox([add, remove]),
                w.HBox([self.window, self.top_k]),
                self.use_all,
                self.combine,
                w.Accordion(
                    children=[w.VBox([self.max_words, self.temperature])],
                    titles=("Advanced settings",),
                    selected_index=None,
                ),
                validate,
                self.status,
                self.preview,
            ]
        )
        if construct:
            for component in construct.components:
                self.add_component(component)
        else:
            self.add_component()

    def add_component(self, component=None):
        w = self._w
        fields = {
            "name": w.Text(
                value=component.name if component else f"component_{len(self.fields) + 1}",
                description="Component",
            ),
            "description": w.Text(
                value=component.description if component else "", description="Meaning"
            ),
        }
        for name, label in [
            ("positive", "Positive"),
            ("contrast", "Contrast"),
            ("hypotheses", "NLI claims"),
        ]:
            fields[name] = w.Textarea(
                value="\n".join(getattr(component, name)) if component else "",
                description=label,
                placeholder="One proposition per line",
                layout=w.Layout(width="100%", height="130px"),
            )
        fields["box"] = w.VBox(list(fields.values()))
        self.fields.append(fields)
        self._refresh()

    def remove_component(self):
        if len(self.fields) > 1:
            self.fields.pop(self.tabs.selected_index or 0)
            self._refresh()

    def _refresh(self):
        self.tabs.children = tuple(f["box"] for f in self.fields)
        for i, f in enumerate(self.fields):
            self.tabs.set_title(i, f["name"].value or f"Component {i + 1}")

    def build(self) -> Construct:
        components = []
        for fields in self.fields:
            values = {
                key: tuple(line.strip() for line in fields[key].value.splitlines() if line.strip())
                for key in ("positive", "contrast", "hypotheses")
            }
            components.append(
                Component(
                    fields["name"].value.strip(), description=fields["description"].value, **values
                )
            )
        return Construct(
            self.name.value.strip(),
            components,
            window=self.window.value,
            top_k=None if self.use_all.value else self.top_k.value,
            max_words=self.max_words.value,
            combine=self.combine.value,
            temperature=self.temperature.value,
        )

    def _preview(self):
        try:
            construct = self.build()
            self.preview.value = json.dumps(construct.to_dict(), indent=2, ensure_ascii=False)
            self.status.value = f"<p>Valid specification: {len(construct.components)} component(s). Call <code>editor.build()</code> to use it in Python. Validation here checks the specification, not measurement validity.</p>"
            self._refresh()
        except (ValueError, TypeError) as error:
            self.preview.value = ""
            self.status.value = (
                "<p><strong>Check the specification:</strong> " + html.escape(str(error)) + "</p>"
            )

    def _ipython_display_(self):
        from IPython.display import display

        display(self.widget)


def construct_editor(construct=None) -> ConstructEditor:
    """Return a notebook form; display(editor), then editor.build() after editing."""
    return ConstructEditor(construct)
