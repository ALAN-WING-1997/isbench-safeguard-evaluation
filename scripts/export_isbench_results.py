"""Export compact evidence and bounded archive parts without rerunning simulation."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


COMPACT_NAMES = {
    "batch_summary.json", "batch_summary_corrected.json", "batch_manifest.json",
    "model_metadata.json", "cases.json", "tasks.txt", "pilot_tasks.txt",
    "summary.json", "summary_corrected.json", "report.json", "task_config.json",
    "progress.json", "process_exit.json", "cleanup_started.txt", "cleanup_completed.txt",
    "initial_state.json", "runtime.json", "case.json",
}


def export_results(run_dir, part_size_bytes=100 * 1024 * 1024):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or not (run_dir / "batch_summary.json").is_file():
        raise RuntimeError("Expected a saved model batch folder with batch_summary.json: " + str(run_dir))
    if part_size_bytes <= 0:
        raise ValueError("Part size must be positive.")
    export_dir = run_dir.parent / (run_dir.name + "_downloads")
    export_dir.mkdir(exist_ok=True)
    summary_zip = export_dir / (run_dir.name + "_summary.zip")
    temporary = summary_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and (path.name in COMPACT_NAMES or
                                  (path.suffix == ".py" and
                                   (path.parent == run_dir or "code" in path.relative_to(run_dir).parts))):
                archive.write(path, Path(run_dir.name) / path.relative_to(run_dir))
    temporary.replace(summary_zip)
    print("COMPACT_REPORT_ZIP:", summary_zip,
          f"({summary_zip.stat().st_size / 1024**2:.2f} MiB)", flush=True)

    full_zip = run_dir.parent / (run_dir.name + ".zip")
    newest_artifact = max(path.stat().st_mtime for path in run_dir.rglob("*") if path.is_file())
    if not full_zip.is_file() or full_zip.stat().st_mtime < newest_artifact:
        # A partial batch may have been exported before later episodes finished.
        # Keep that archive and build a current snapshot under the export folder.
        full_zip = export_dir / (run_dir.name + "_full.zip")
    if not full_zip.is_file() or full_zip.stat().st_mtime < newest_artifact:
        print("Creating the full result archive...", flush=True)
        temporary_base = export_dir / (run_dir.name + "_building")
        created = Path(shutil.make_archive(str(temporary_base), "zip",
                                          root_dir=run_dir.parent, base_dir=run_dir.name))
        created.replace(full_zip)
    if not zipfile.is_zipfile(full_zip):
        raise RuntimeError("The existing full archive is not a ZIP file: " + str(full_zip))

    # Existing full ZIP and raw artifacts are preserved. Each transfer is bounded;
    # never read or encode the full archive in a single operation.
    full_digest = hashlib.sha256()
    parts = []
    with full_zip.open("rb") as stream:
        while True:
            chunk = stream.read(part_size_bytes)
            if not chunk:
                break
            part = export_dir / f"{full_zip.name}.part{len(parts)+1:03d}"
            temporary = part.with_suffix(part.suffix + ".tmp")
            temporary.write_bytes(chunk)
            temporary.replace(part)
            full_digest.update(chunk)
            parts.append({"file": part.name, "bytes": len(chunk),
                          "sha256": hashlib.sha256(chunk).hexdigest()})
            print("FULL_ARCHIVE_PART:", part, f"({len(chunk) / 1024**2:.2f} MiB)", flush=True)
    manifest = {"archive_name": full_zip.name, "archive_bytes": full_zip.stat().st_size,
                "archive_sha256": full_digest.hexdigest(), "part_size_bytes": part_size_bytes,
                "parts": parts, "summary_zip": summary_zip.name,
                "note": "Concatenate parts in the listed order to restore the original ZIP. Parts are not independent ZIPs."}
    manifest_path = export_dir / "download_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)
    return {"summary_zip": str(summary_zip), "full_zip": str(full_zip),
            "export_dir": str(export_dir), "manifest": str(manifest_path),
            "parts": [str(export_dir / part["file"]) for part in parts]}
