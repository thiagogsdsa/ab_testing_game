import shutil
import subprocess
from pathlib import Path
import nbformat
import tempfile

def set_project_root(notebook_path: str, root_folder_name: str = None):
    notebook_path = Path(notebook_path).resolve()
    if root_folder_name:
        for parent in notebook_path.parents:
            if parent.name == root_folder_name:
                return parent
        raise ValueError(f"Root folder '{root_folder_name}' not found in notebook path.")
    else:
        # Default: one level above 'notebooks' folder
        return notebook_path.parent.parent

def fix_image_paths_in_copy(original_path: Path, imgs_folder: str = "imgs"):
    """
    Make a temporary copy of the notebook and fix image paths for HTML.
    Returns path to the temporary notebook copy.
    """
    nb = nbformat.read(original_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace("../imgs/", f"{imgs_folder}/")

    tmp_dir = tempfile.mkdtemp()
    tmp_notebook_path = Path(tmp_dir) / original_path.name
    nbformat.write(nb, tmp_notebook_path)
    return tmp_notebook_path

def convert_notebook_to_index(
    notebook_path: str,
    output_name: str = "index.html",
    execute: bool = True,
    timeout: int = 600
):
    notebook_path = Path(notebook_path).resolve()
    project_root = set_project_root(notebook_path)
    output_path = project_root / output_name

    # --- Make a safe copy and fix image paths ---
    tmp_notebook = fix_image_paths_in_copy(notebook_path)

    # --- Build command ---
    cmd = [
        "jupyter", "nbconvert",
        "--to", "html",
        str(tmp_notebook.name),
        "--output", output_name
    ]
    if execute:
        cmd.insert(2, "--execute")
        cmd.insert(3, f"--ExecutePreprocessor.timeout={timeout}")

    # --- Run nbconvert in temporary folder ---
    try:
        subprocess.run(cmd, cwd=tmp_notebook.parent, check=True)
    except subprocess.CalledProcessError:
        print("Execution failed. Generating HTML without executing...")
        cmd_no_exec = [c for c in cmd if c != "--execute" and not c.startswith("--ExecutePreprocessor")]
        subprocess.run(cmd_no_exec, cwd=tmp_notebook.parent, check=True)

    # --- Move HTML to project root ---
    generated_html = tmp_notebook.parent / output_name
    if generated_html.exists():
        shutil.move(str(generated_html), str(output_path))

    print(f" HTML generated safely at: {output_path.resolve()}")
    # Optional: clean temp folder
    shutil.rmtree(tmp_notebook.parent, ignore_errors=True)

# --- USAGE ---
notebook_path = "notebooks/1.0.tgs.cookies_cats_ab_test.ipynb"
convert_notebook_to_index(notebook_path)