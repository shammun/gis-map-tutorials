"""Turn the tutorial scripts (percent format: ``# %%`` cells) into notebooks.

Every tutorial is written once, as ``tutorials/<group>/NN_name.py``. For each one this writes

* ``NN_name.ipynb``        the notebook (then executed in place with nbconvert, outputs kept);
* ``NN_name_colab.ipynb``  the Google Colab twin: no outputs, and in front a first cell with the "Open in Colab"
  badge and a setup cell that ``pip install``s this repository's package. No keys, no local paths.

    python tools/build_notebooks.py                 # all ten, executed
    python tools/build_notebooks.py gis_styles/03   # the ones whose path contains this text
    python tools/build_notebooks.py --no-exec       # write without executing

Banners. A step heading (``## N · Title`` in a markdown cell) of the style tutorials, and a code cell that starts with
``# %% STATUS :: Title :: one line`` in the lens tutorials, become a coloured box: green = run it, orange = a
setting you may change, blue / grey = optional or conditional.

Edit the ``.py`` files, never the notebooks.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KERNEL = os.environ.get("GISMAPS_KERNEL", "python3")

GREEN, ORANGE, BLUE, GREY, RED = (("#E8F5E9", "#2E7D32"), ("#FFF3E0", "#E65100"), ("#E3F2FD", "#1565C0"),
                                  ("#ECEFF1", "#546E7A"), ("#FFEBEE", "#C62828"))
HEADING_RULES = [                                  # (words in the step title, colours, status line)
    (("setup (google colab)",), GREEN, "RUN FIRST"),
    (("settings",), ORANGE, "CHANGE IF YOU LIKE — RUN"),
    (("exporting",), BLUE, "OPTIONAL — NEEDS USE_GEE = True; RUN THE CHECK AFTER THE EXPORTS FINISH"),
    (("earth engine",), BLUE, "OPTIONAL — NEEDS AN EARTH ENGINE ACCOUNT; DOES NOTHING WHILE USE_GEE = False"),
    (("animation",), GREEN, "RUN — A SHORT PREVIEW IN FAST MODE, THE WHOLE ANIMATION IN FULL MODE"),
    (("check the outputs",), GREEN, "RUN — PASS OR FAIL FOR EVERY OUTPUT"),
    (("download",), BLUE, "ON GOOGLE COLAB — ONE ZIP OF THE OUTPUTS"),
]
STATUS = {"RUN": GREEN, "CHANGE": ORANGE, "OPTIONAL": GREY, "STOP": RED}


def box(status: str, title: str, colours, line: str = "") -> str:
    bg, fg = colours
    return (f'<div style="background:{bg};border:3px solid {fg};border-left:14px solid {fg};border-radius:6px;'
            f'padding:12px 16px;margin:8px 0;color:#1b1b1b">\n'
            f'<div style="color:{fg};font-size:19px;font-weight:bold">{status}</div>\n'
            f'<div style="font-size:16px;font-weight:bold;margin-top:4px">{title}</div>\n'
            + (f'<div style="font-size:14px;margin-top:4px">{line}</div>\n' if line else "") + "</div>")


def heading_banner(markdown: str) -> str:
    """``## N · Title`` + text -> a coloured banner, then the text."""
    first, _, rest = markdown.partition("\n")
    if not first.startswith("## "):
        return markdown
    title = first[3:].strip()
    colours, status = GREEN, "RUN"
    for words, c, text in HEADING_RULES:
        if any(w in title.lower() for w in words):
            colours, status = c, text
            break
    return box(status, title, colours) + ("\n\n" + rest.strip("\n") if rest.strip() else "")


def spec_banner(spec: str) -> str:
    """``STATUS :: Title :: one line`` -> the coloured banner shown above a code cell."""
    status, title, line = (part.strip() for part in (spec.split("::") + ["", ""])[:3])
    return box(status, title, STATUS.get(status.split()[0], GREEN), line)


def cells_of(script: Path, heading_banners: bool) -> list:
    """Split a percent-format script into notebook cells."""
    out, kind, buf, spec = [], None, [], ""

    def flush():
        body = "\n".join(buf).strip("\n")
        if not kind or not body.strip():
            return
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.split("\n"))
            out.append(nbf.v4.new_markdown_cell(heading_banner(body) if heading_banners else body))
        else:
            if spec:
                out.append(nbf.v4.new_markdown_cell(spec_banner(spec)))
            out.append(nbf.v4.new_code_cell(body))

    for line in script.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
            spec = line[4:].strip() if kind == "code" else ""
        elif kind:
            buf.append(line)
    flush()
    return out


def first_cell(stem: str, colab: bool) -> nbf.NotebookNode:
    """Badge, one-line description and how to run."""
    t = catalog.TUTORIALS[stem]
    name = Path(stem).name
    if colab:
        how = ("**Tap ▸ Runtime ▸ Run all.** No key and no account are needed. The first code cell installs the packages "
               "(about a minute); the pictures and videos appear in the notebook, are written to `/content/out`, and the "
               "last cell offers them as one download. To keep your changes: File ▸ Save a copy in Drive.")
    else:
        how = (f"This is the executed notebook (outputs kept). The same recipe as a script: `{name}.py`; "
               f"for Google Colab: `{name}_colab.ipynb` (the badge opens it).")
    page = f"{catalog.PAGES}/tutorials/{name}.html"
    return nbf.v4.new_markdown_cell(
        f"{catalog.badge(stem)}\n\n**{t['title']}.** {t['one']}\n\n{how}\n\n"
        f"[Gallery]({catalog.PAGES}/) · [this recipe as a web page]({page}) · [repository]({catalog.GITHUB}) · "
        f"{catalog.AUTHOR}")


def colab_setup(stem: str) -> list:
    extra = catalog.TUTORIALS[stem]["pip"]
    code = ("import os\n\n"
            "# This repository's package (gismaps) and what it needs. numpy, matplotlib, pandas and shapely are already in\n"
            "# Colab and are left as they are, so no restart of the runtime is needed.\n"
            f'GISMAPS = os.environ.get("GISMAPS_PIP_SOURCE", "git+{catalog.GITHUB}")   # (the variable only exists so a local copy can be tested)\n'
            f'%pip install -q "$GISMAPS"{" " + extra if extra else ""}\n\n'
            "import gismaps\n"
            'print("gismaps", gismaps.__version__, "is installed")')
    text = ("## Setup (Google Colab)\n"
            "One cell, no key and no account: it installs this repository's small package `gismaps` (the code behind the "
            "maps) together with the packages it needs" + (f", plus `{extra}` for this tutorial" if extra else "") +
            ". The small input files are downloaded from the repository the first time they are used.")
    return [nbf.v4.new_markdown_cell(box("RUN FIRST", "Setup (Google Colab)", GREEN) + "\n\n" + text.split("\n", 1)[1]),
            nbf.v4.new_code_cell(code)]


def static_widgets(nb) -> None:
    """A saved notebook cannot keep a live panel: replace widget outputs by one line of text and drop the widget state."""
    nb.metadata.pop("widgets", None)
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            data = o.get("data", {})
            if "application/vnd.jupyter.widget-view+json" in data:
                data.pop("application/vnd.jupyter.widget-view+json")
                data["text/plain"] = "(the panel appears here when the notebook runs)"


def build(script: Path, execute: bool = True) -> float | None:
    stem = f"{script.parent.name}/{script.stem}"
    styled = script.parent.name != "lens_maps"          # lens recipes carry their banners on the code cells
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    nb = nbf.v4.new_notebook(cells=[first_cell(stem, False), *cells_of(script, styled)], metadata=meta)
    out = script.with_suffix(".ipynb")
    nbf.write(nb, out)
    twin = nbf.v4.new_notebook(cells=[first_cell(stem, True), *colab_setup(stem), *cells_of(script, styled)],
                               metadata={**meta, "colab": {"provenance": [], "toc_visible": True}})
    nbf.write(twin, script.with_name(script.stem + "_colab.ipynb"))
    took = None
    if execute:
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=3600", f"--ExecutePreprocessor.kernel_name={KERNEL}", str(out)],
                       check=True, cwd=script.parent, env=env)
        took = time.time() - t0
        done = nbf.read(out, as_version=4)          # keep the generic kernel name in the file, whatever ran it
        done.metadata["kernelspec"] = meta["kernelspec"]
        static_widgets(done)
        nbf.write(done, out)
    print(f"built {out.relative_to(ROOT)} and its Colab twin" + (f" | executed in {took:.0f} s" if took else ""))
    return took


if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    scripts = sorted(p for p in (ROOT / "tutorials").glob("*/[0-9][0-9]_*.py")
                     if not wanted or any(w in p.as_posix() for w in wanted))
    for s in scripts:
        build(s, execute="--no-exec" not in sys.argv)
