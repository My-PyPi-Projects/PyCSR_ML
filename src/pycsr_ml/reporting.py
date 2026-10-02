"""HTML report rendering."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Union

from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_report(context: dict, output_path: Union[str, Path]) -> Path:
    template_dir = files("pycsr_ml").joinpath("templates")
    environment = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    environment.filters["comma"] = lambda value: f"{value:,}"
    template = environment.get_template("report.html")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.render(**context), encoding="utf-8")
    return destination
