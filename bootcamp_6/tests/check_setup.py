"""Validate both notebook graphs, solution results and saved-answer preservation."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    subprocess.run([sys.executable, str(ROOT / "tests/check_launchers.py")], check=True)
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    for folder in ["exercises", "solutions"]:
        subprocess.run([sys.executable, "-m", "marimo", "check", str(ROOT / folder / "workbook.py")], check=True)
    student = load(ROOT / "exercises/workbook.py", "student")
    _, blank = student.app.run()
    assert "X_train" in blank and "test_results" not in blank
    plt.close("all")
    solution = load(ROOT / "solutions/workbook.py", "solution")
    _, results = solution.app.run()
    assert [len(results[k]) for k in ["y_train", "y_val", "y_test"]] == [90, 30, 30]
    for name in ["y_train", "y_val", "y_test"]:
        counts = np.bincount(results[name])
        assert len(set(counts)) == 1, "The stratified split should keep equal class counts."
    np.testing.assert_allclose(results["knn_model"].named_steps["standardscaler"].mean_, results["X_train"].mean(axis=0))
    assert results["tree_model"].get_depth() <= 2
    assert accuracy_score(results["y_test"], results["knn_test_predictions"]) >= 0.8
    assert accuracy_score(results["y_test"], results["tree_test_predictions"]) >= 0.8
    assert list(results["knn_results"]["k"]) == [1, 5, 15, 30]
    assert len(results["tree_results"]) == 5
    # Unit changes should disappear after fitting a scaler on training data.
    changed = results["X_train"].copy()
    changed[:, 0] *= 10000
    np.testing.assert_allclose(StandardScaler().fit_transform(changed),
                               StandardScaler().fit_transform(results["X_train"]), atol=1e-12)
    print(results["test_results"].to_string(index=False))
    plt.close("all")

    manager = load(ROOT / "scripts/manage.py", "manager")
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        for folder in ["exercises", "solutions"]:
            (temp / folder).mkdir()
            (temp / folder / "workbook.py").write_text(folder + " template")
        personal = manager.working_copy(root=temp)
        personal.write_text("student answers")
        (temp / "exercises/workbook.py").write_text("updated template")
        assert manager.working_copy(root=temp).read_text() == "student answers"
        teacher = manager.working_copy(solution=True, root=temp)
        assert teacher != personal and teacher.read_text() == "solutions template"
        teacher.write_text("teacher notes")
        assert manager.working_copy(solution=True, root=temp).read_text() == "teacher notes"
    subprocess.run([sys.executable, str(ROOT.parent / "scripts/check_repository.py")], check=True)
    print("PASS: blank workbook, full solutions, train-only scaling and saved-answer preservation.")


if __name__ == "__main__":
    main()
