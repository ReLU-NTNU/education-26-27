# ReLU education 2026–2027

Each bootcamp has its own folder and setup. For today’s KNN and decision-tree
workbook, follow [Bootcamp 6](bootcamp_6/README.md). Existing Bootcamp 2 users can
keep using their current commands and saved work.

## Start a bootcamp

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/ReLU-NTNU/education-26-27.git
cd education-26-27/bootcamp_2
```

| | macOS / Linux | Windows PowerShell |
|---|---|---|
| Set up once | `./scripts/setup.sh` | `.\scripts\setup.ps1` |
| Start or resume | `./scripts/start.sh` | `.\scripts\start.ps1` |

The launcher opens your personal notebook in marimo. It preserves your answers
when you restart or update the course.

- [Bootcamp 2: Python for machine learning](bootcamp_2/README.md)
- [Bootcamp 6: KNN and single decision trees](bootcamp_6/README.md) — marimo exercises and solutions.
- Teaching templates and solutions are versioned in Git.
- Personal answers live in each bootcamp's ignored `work/` directory.
- Datasets download on demand to your OS cache **outside this repository**.
- Python environments and generated files are local and excluded from Git.

Maintainers: run `python scripts/check_repository.py` before committing. It
rejects tracked datasets, environments, student work, model files and archives.
Each bootcamp also runs this check as part of its test command.
