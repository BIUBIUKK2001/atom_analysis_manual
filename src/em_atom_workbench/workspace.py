from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import subprocess

import pandas as pd

from .utils import serializable, slugify, stage_rank, write_json

WORKSPACE_SCHEMA_VERSION = "1.0"

CANONICAL_STAGE_DIRS = {
    "01_findatom": "01_findatom",
    "02_simple_quant": "02_simple_quant",
    "03_group_centroid": "03_group_centroid",
}

STAGE_DIR_ALIASES = {
    "01": "01_findatom",
    "findatom": "01_findatom",
    "01_findatom": "01_findatom",
    "02": "02_simple_quant",
    "simple_quant": "02_simple_quant",
    "simple_quant_v2": "02_simple_quant",
    "02_simple_quant": "02_simple_quant",
    "03": "03_group_centroid",
    "group_centroid": "03_group_centroid",
    "cropped_group_centroid": "03_group_centroid",
    "03_group_centroid": "03_group_centroid",
    "03_cropped_group_centroid": "03_group_centroid",
}

CANONICAL_STAGE_SESSIONS = {
    "01_loaded",
    "01_candidate_reviewed",
    "01_class_reviewed",
    "01_refined",
    "01_final_curated",
    "02_simple_quant",
    "03_group_centroid",
}

STAGE_SESSION_ALIASES = {
    "01_loaded": "01_loaded",
    "01_candidate_reviewed": "01_candidate_reviewed",
    "01_class_reviewed": "01_class_reviewed",
    "01_refined": "01_refined",
    "01_final_curated": "01_final_curated",
    "02_simple_quant": "02_simple_quant",
    "simple_quant": "02_simple_quant",
    "simple_quant_v2": "02_simple_quant",
    "03_group_centroid": "03_group_centroid",
    "group_centroid": "03_group_centroid",
    "cropped_group_centroid": "03_group_centroid",
    "03_cropped_group_centroid": "03_group_centroid",
}

STAGE_SUBDIRS = {
    "01_findatom": ("configs", "tables", "figures_preview", "figures_final", "checkpoints"),
    "02_simple_quant": ("configs", "tables", "figures_preview", "figures_final", "session"),
    "03_group_centroid": ("configs", "tables", "figures_preview", "figures_final", "session"),
}


@dataclass
class AnalysisWorkspace:
    output_root: Path
    dataset_id: str
    analysis_id: str
    root: Path
    state_dir: Path
    sessions_dir: Path
    shared_dir: Path
    manifests_dir: Path
    stage_dirs: dict[str, Path]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _normalize_stage_dir_name(stage_name: str) -> str:
    key = str(stage_name).strip()
    normalized = STAGE_DIR_ALIASES.get(key)
    if normalized is None:
        raise KeyError(
            f"Unknown stage folder {stage_name!r}. Expected one of: "
            + ", ".join(sorted(CANONICAL_STAGE_DIRS))
        )
    return normalized


def _normalize_stage_session_name(stage_name: str) -> str:
    key = str(stage_name).strip()
    normalized = STAGE_SESSION_ALIASES.get(key)
    if normalized is None or normalized not in CANONICAL_STAGE_SESSIONS:
        raise KeyError(
            f"Unknown stage session {stage_name!r}. Expected one of: "
            + ", ".join(sorted(CANONICAL_STAGE_SESSIONS))
        )
    return normalized


def initialize_analysis_workspace(
    output_root: str | Path,
    dataset_id: str,
    analysis_id: str,
    *,
    create: bool = True,
) -> AnalysisWorkspace:
    output_root_path = Path(output_root)
    root = output_root_path / slugify(str(dataset_id)) / slugify(str(analysis_id))
    workspace = AnalysisWorkspace(
        output_root=output_root_path,
        dataset_id=str(dataset_id),
        analysis_id=str(analysis_id),
        root=root,
        state_dir=root / "state",
        sessions_dir=root / "state" / "sessions",
        shared_dir=root / "shared",
        manifests_dir=root / "manifests",
        stage_dirs={name: root / name for name in CANONICAL_STAGE_DIRS},
    )
    if create:
        _create_workspace_dirs(workspace)
        _write_project_config(workspace)
        shared_files = [
            workspace.shared_dir / "channel_summary.csv",
            workspace.shared_dir / "pixel_calibration.json",
            workspace.shared_dir / "input_metadata.json",
        ]
        if any(not path.exists() for path in shared_files):
            write_shared_workspace_files(workspace, session=None)
    return workspace


def _create_workspace_dirs(workspace: AnalysisWorkspace) -> None:
    base_dirs = [
        workspace.root,
        workspace.state_dir,
        workspace.sessions_dir,
        workspace.shared_dir,
        workspace.manifests_dir,
    ]
    for path in base_dirs:
        path.mkdir(parents=True, exist_ok=True)
    for stage_name, stage_dir in workspace.stage_dirs.items():
        stage_dir.mkdir(parents=True, exist_ok=True)
        for subdir in STAGE_SUBDIRS[stage_name]:
            (stage_dir / subdir).mkdir(parents=True, exist_ok=True)


def _write_project_config(workspace: AnalysisWorkspace) -> Path:
    config_path = workspace.root / "project_config.json"
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}
    now = _now_iso()
    payload = dict(existing)
    payload.update(
        {
            "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
            "output_root": str(workspace.output_root),
            "dataset_id": workspace.dataset_id,
            "analysis_id": workspace.analysis_id,
            "root": str(workspace.root),
            "stage_dirs": {name: str(path) for name, path in workspace.stage_dirs.items()},
            "state_dir": str(workspace.state_dir),
            "sessions_dir": str(workspace.sessions_dir),
            "shared_dir": str(workspace.shared_dir),
            "manifests_dir": str(workspace.manifests_dir),
            "created_at": payload.get("created_at", now),
            "updated_at": now,
        }
    )
    git_commit = _safe_git_commit(Path.cwd())
    if git_commit is not None:
        payload["git_commit"] = git_commit
    return write_json(config_path, payload)


def get_stage_dir(workspace: AnalysisWorkspace, stage_name: str) -> Path:
    canonical = _normalize_stage_dir_name(stage_name)
    return workspace.stage_dirs[canonical]


def get_stage_subdir(workspace: AnalysisWorkspace, stage_name: str, subdir: str) -> Path:
    target = get_stage_dir(workspace, stage_name) / str(subdir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def stage_session_path(workspace: AnalysisWorkspace, stage_name: str) -> Path:
    canonical = _normalize_stage_session_name(stage_name)
    return workspace.sessions_dir / f"{canonical}.pkl"


def _session_manifest(session: Any) -> dict[str, Any]:
    if hasattr(session, "to_manifest_dict"):
        try:
            return dict(session.to_manifest_dict())
        except Exception:
            pass
    return {
        "name": getattr(session, "name", None),
        "input_path": getattr(session, "input_path", None),
        "dataset_index": getattr(session, "dataset_index", None),
        "current_stage": getattr(session, "current_stage", "loaded"),
    }


def write_shared_workspace_files(workspace: AnalysisWorkspace, session: Any | None = None) -> dict[str, Path]:
    workspace.shared_dir.mkdir(parents=True, exist_ok=True)
    if session is None:
        channel_table = pd.DataFrame(columns=["channel", "is_primary", "input_path", "dataset_index", "contrast_mode", "raw_shape"])
        pixel_payload: dict[str, Any] = {}
        metadata_payload: dict[str, Any] = {}
    else:
        channels = []
        channel_metadata: dict[str, Any] = {}
        try:
            channel_names = session.list_channels()
        except Exception:
            channel_names = []
        for channel_name in channel_names:
            try:
                state = session.get_channel_state(channel_name)
                raw_shape = None if state.raw_image is None else tuple(state.raw_image.shape)
                channels.append(
                    {
                        "channel": channel_name,
                        "is_primary": channel_name == getattr(session, "primary_channel", None),
                        "input_path": state.input_path,
                        "dataset_index": state.dataset_index,
                        "contrast_mode": state.contrast_mode,
                        "raw_shape": raw_shape,
                    }
                )
                channel_metadata[channel_name] = serializable(getattr(state, "raw_metadata", {}) or {})
            except Exception:
                channels.append({"channel": channel_name})
        channel_table = pd.DataFrame(channels)
        pixel_payload = serializable(getattr(session, "pixel_calibration", {}))
        metadata_payload = {
            "session_name": getattr(session, "name", None),
            "input_path": getattr(session, "input_path", None),
            "dataset_index": getattr(session, "dataset_index", None),
            "raw_metadata": serializable(getattr(session, "raw_metadata", {}) or {}),
            "channel_metadata": channel_metadata,
        }
    channel_path = workspace.shared_dir / "channel_summary.csv"
    channel_table.to_csv(channel_path, index=False)
    pixel_path = write_json(workspace.shared_dir / "pixel_calibration.json", pixel_payload)
    metadata_path = write_json(workspace.shared_dir / "input_metadata.json", metadata_payload)
    return {
        "channel_summary": channel_path,
        "pixel_calibration": pixel_path,
        "input_metadata": metadata_path,
    }


def save_stage_session(
    session: Any,
    workspace: AnalysisWorkspace,
    stage_name: str,
    *,
    update_active: bool = True,
    notes: dict | None = None,
) -> Path:
    canonical = _normalize_stage_session_name(stage_name)
    target = stage_session_path(workspace, canonical)
    saved_path = session.save_pickle(target)
    write_shared_workspace_files(workspace, session=session)
    if update_active:
        active_path = session.save_pickle(workspace.state_dir / "active_session.pkl")
        updated_at = _now_iso()
        active_payload = {
            "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
            "session_name": getattr(session, "name", None),
            "current_stage": getattr(session, "current_stage", "loaded"),
            "stage_name": canonical,
            "stage_session_path": str(saved_path),
            "active_session_path": str(active_path),
            "dataset_id": workspace.dataset_id,
            "analysis_id": workspace.analysis_id,
            "updated_at": updated_at,
            "notes": notes or {},
        }
        write_json(workspace.state_dir / "active_session.json", active_payload)
        _update_latest_session(workspace, canonical, saved_path, active_path, session, notes=notes)
    return saved_path


def _update_latest_session(
    workspace: AnalysisWorkspace,
    stage_name: str,
    stage_path: Path,
    active_path: Path,
    session: Any,
    *,
    notes: dict | None = None,
) -> Path:
    latest_path = workspace.state_dir / "latest_session.json"
    existing: dict[str, Any] = {}
    if latest_path.exists():
        try:
            loaded = json.loads(latest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}
    history = list(existing.get("stage_history", []))
    updated_at = _now_iso()
    history.append(
        {
            "stage_name": stage_name,
            "session_name": getattr(session, "name", None),
            "current_stage": getattr(session, "current_stage", "loaded"),
            "stage_session_path": str(stage_path),
            "active_session_path": str(active_path),
            "updated_at": updated_at,
            "notes": notes or {},
        }
    )
    payload = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "dataset_id": workspace.dataset_id,
        "analysis_id": workspace.analysis_id,
        "latest_stage_name": stage_name,
        "latest_session_path": str(stage_path),
        "active_session_path": str(active_path),
        "updated_at": updated_at,
        "stage_history": history,
    }
    return write_json(latest_path, payload)


def _ensure_required_stage_for_workspace(session: Any, required_stage: str | None, source_label: str) -> None:
    if required_stage is None:
        return
    current = getattr(session, "current_stage", "loaded")
    current_rank = stage_rank(current)
    required_rank = stage_rank(required_stage)
    if current_rank < 0 or required_rank < 0:
        return
    if current_rank < required_rank:
        raise ValueError(
            f"{source_label} stage '{current}' does not satisfy the required stage '{required_stage}'."
        )


def load_stage_session(
    workspace: AnalysisWorkspace,
    stage_name: str | None = "01_final_curated",
    *,
    session_path: str | Path | None = None,
    required_stage: str | None = None,
) -> Any:
    from .session import AnalysisSession

    if session_path is not None:
        path = Path(session_path)
        if not path.exists():
            raise FileNotFoundError(f"Session path does not exist: {path}")
        source_label = f"Session path {path}"
    else:
        canonical = _normalize_stage_session_name(stage_name or "01_final_curated")
        path = stage_session_path(workspace, canonical)
        source_label = f"Stage session '{canonical}'"
        if not path.exists():
            raise FileNotFoundError(
                f"Stage session '{canonical}' was not found. Run notebook 01 and save final session first, "
                "or set SESSION_PATH manually."
            )
    session = AnalysisSession.load_pickle(path)
    _ensure_required_stage_for_workspace(session, required_stage, source_label)
    return session


def load_active_workspace_session(workspace: AnalysisWorkspace, required_stage: str | None = None) -> Any:
    from .session import AnalysisSession

    active_path = workspace.state_dir / "active_session.pkl"
    if not active_path.exists():
        raise FileNotFoundError(
            "Workspace active session was not found. Run a notebook stage first, or set SESSION_PATH manually."
        )
    session = AnalysisSession.load_pickle(active_path)
    _ensure_required_stage_for_workspace(session, required_stage, "Workspace active session")
    return session


def write_workspace_json(path: str | Path, payload: dict[str, Any]) -> Path:
    return write_json(path, payload)


def export_stage_table(
    workspace: AnalysisWorkspace,
    stage_name: str,
    table_name: str,
    table: Any,
    *,
    subdir: str = "tables",
    formats: tuple[str, ...] = ("csv",),
) -> list[Path]:
    target_dir = get_stage_subdir(workspace, stage_name, subdir)
    data = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    paths: list[Path] = []
    for fmt in formats:
        suffix = str(fmt).lower().lstrip(".")
        target = target_dir / f"{table_name}.{suffix}"
        if suffix == "csv":
            data.to_csv(target, index=False)
        elif suffix in {"xlsx", "xls"}:
            data.to_excel(target, index=False)
        else:
            raise ValueError(f"Unsupported table export format: {fmt!r}")
        paths.append(target)
    return paths


def export_stage_figure(
    workspace: AnalysisWorkspace,
    stage_name: str,
    figure_name: str,
    fig: Any,
    *,
    final: bool = True,
    formats: tuple[str, ...] = ("pdf", "png", "svg"),
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> list[Path]:
    subdir = "figures_final" if final else "figures_preview"
    target_dir = get_stage_subdir(workspace, stage_name, subdir)
    paths: list[Path] = []
    for fmt in formats:
        suffix = str(fmt).lower().lstrip(".")
        target = target_dir / f"{figure_name}.{suffix}"
        fig.savefig(target, dpi=int(dpi), bbox_inches=bbox_inches)
        paths.append(target)
    return paths


def write_stage_manifest(
    workspace: AnalysisWorkspace,
    stage_name: str,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    canonical = _normalize_stage_dir_name(stage_name)
    now = _now_iso()
    payload = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "dataset_id": workspace.dataset_id,
        "analysis_id": workspace.analysis_id,
        "stage_name": canonical,
        "updated_at": now,
    }
    payload.update(dict(manifest or {}))
    payload.setdefault("exported_at", now)
    payload.setdefault("tables", {})
    payload.setdefault("figures", {})
    payload.setdefault("configs", {})
    payload.setdefault("session_paths", {})
    stage_manifest = write_json(get_stage_dir(workspace, canonical) / "manifest.json", payload)
    root_manifest = write_json(workspace.manifests_dir / f"{canonical}_manifest.json", payload)
    return {"stage_manifest": stage_manifest, "workspace_manifest": root_manifest}


def collect_project_manifest(workspace: AnalysisWorkspace) -> Path:
    stages: dict[str, Any] = {}
    for stage_name in CANONICAL_STAGE_DIRS:
        stage_manifest = get_stage_dir(workspace, stage_name) / "manifest.json"
        workspace_manifest = workspace.manifests_dir / f"{stage_name}_manifest.json"
        stages[stage_name] = {
            "stage_manifest": str(stage_manifest) if stage_manifest.exists() else None,
            "workspace_manifest": str(workspace_manifest) if workspace_manifest.exists() else None,
        }
    session_paths = {
        stage_name: str(stage_session_path(workspace, stage_name))
        for stage_name in sorted(CANONICAL_STAGE_SESSIONS)
    }
    payload = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "dataset_id": workspace.dataset_id,
        "analysis_id": workspace.analysis_id,
        "updated_at": _now_iso(),
        "root": str(workspace.root),
        "stages": stages,
        "session_paths": session_paths,
        "active_session_path": str(workspace.state_dir / "active_session.pkl"),
    }
    return write_json(workspace.manifests_dir / "project_manifest.json", payload)
