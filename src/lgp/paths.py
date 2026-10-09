from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root without depending on the current directory."""
    configured = os.environ.get("LGP_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "configs" / "models.yaml").is_file():
            raise FileNotFoundError(
                "LGP_PROJECT_ROOT does not contain configs/models.yaml: {}".format(root)
            )
        return root

    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    seen = set()
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            resolved = parent.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if (resolved / "configs" / "models.yaml").is_file():
                return resolved
    raise FileNotFoundError(
        "Cannot locate the LGP project root. Set LGP_PROJECT_ROOT explicitly."
    )


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()
