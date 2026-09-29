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
        subprocess.run([sys.executable, "-m", "marimo", "check", "--strict", str(ROOT / folder / "workbook.py")], check=True)
    student = load(ROOT / "exercises/workbook.py", "student")
    _, blank = student.app.run()
    assert "X_train" in blank and "test_results" not in blank
    plt.close("all")
    solution = load(ROOT / "solutions/workbook.py", "solution")
    _, results = solution.app.run()
    # Check the lecture algorithm on unsorted neighbours, all three classes, and a tie.
    predict = results["knn_predict"]
    toy_train = np.array([[10.0], [0.0], [11.0], [1.0], [12.0], [2.0]])
    toy_labels = np.array([2, 0, 2, 0, 1, 1])
    toy_queries = np.array([[0.2], [10.2]])
    np.testing.assert_array_equal(predict(toy_train, toy_labels, toy_queries, k=3), [0, 2])
    np.testing.assert_array_equal(predict(toy_train, toy_labels, toy_queries, k=1), [0, 2])
    tied_labels = np.array([2, 1, 0])
    np.testing.assert_array_equal(predict(np.array([[0.0], [1.0], [2.0]]),
                                          tied_labels, np.array([[0.1]]), k=3), [0])
    from sklearn.neighbors import KNeighborsClassifier
    reference_knn = KNeighborsClassifier(n_neighbors=3)
    reference_knn.fit(results["manual_train"], results["y_train"])
    np.testing.assert_array_equal(results["manual_predictions"],
                                  reference_knn.predict(results["manual_validation"]))
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

    # Fill the actual exercise template, then let marimo rebuild its dependencies
    # just as it does when students edit cells in the notebook editor.
    answers = {
        "neighbour_vote": '"versicolor"',
        "distances": "np.linalg.norm(X_train - flower, axis=1)",
        "nearest": "np.argsort(distances)[:k]",
        "labels": "y_train[nearest]",
        "prediction": "np.argmax(vote_counts)",
        "k_start": "5",
        "knn_fitted": "knn_model.fit(X_train, y_train)",
        "knn_predictions": "knn_model.predict(X_val)",
        "k_values": "[1, 5, 15, 30]",
        "tree_depth": "2",
        "tree_fitted": "tree_model.fit(X_train, y_train)",
        "depth_values": "[1, 2, 3, 5, None]",
        "knn_test_predictions": "final_knn.predict(X_test)",
        "tree_test_predictions": "final_tree.predict(X_test)",
        "coordinate_comparisons": "1000 * 4",
    }
    completed_source = (ROOT / "exercises/workbook.py").read_text()
    for variable, answer in answers.items():
        blank_assignment = f"{variable} = None"
        assert blank_assignment in completed_source, variable
        completed_source = completed_source.replace(blank_assignment, f"{variable} = {answer}")
    with tempfile.TemporaryDirectory() as completed_directory:
        completed_path = Path(completed_directory) / "completed.py"
        completed_path.write_text(completed_source)
        subprocess.run([sys.executable, "-m", "marimo", "check", "--fix", str(completed_path)], check=True)
        subprocess.run([sys.executable, "-m", "marimo", "check", "--strict", str(completed_path)], check=True)
        completed = load(completed_path, "completed_student")
        _, completed_results = completed.app.run()
        np.testing.assert_array_equal(completed_results["manual_predictions"], results["manual_predictions"])
        np.testing.assert_array_equal(completed_results["knn_test_predictions"], results["knn_test_predictions"])
        np.testing.assert_array_equal(completed_results["tree_test_predictions"], results["tree_test_predictions"])
        plt.close("all")
    print("PASS: manual KNN, three-class vote, tie handling and completed student template.")

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
