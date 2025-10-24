from pathlib import Path
import nbformat
from nbconvert import HTMLExporter

def notebook_to_html(notebook_path, output_path):

    notebook_path = Path(notebook_path).resolve()
    output_path = Path(output_path).resolve()

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # Read notebook
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Export to HTML with images embedded
    exporter = HTMLExporter()
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(nb)

    # Save HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    print(f"HTML saved at: {output_path.resolve()}")


    notebook_to_html(
    "../notebooks/1.0.tgs.cookies_cats_ab_test.ipynb",
    "../index.html"
)