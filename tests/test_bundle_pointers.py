import json

from src.modeling.bundle import publish_family_pointers


def test_family_pointer_is_explicit_and_hashes_manifests(tmp_path):
    registry = tmp_path / "bundles"
    version = registry / "abc"
    version.mkdir(parents=True)
    (version / "manifest.json").write_text('{"family":"RF"}', encoding="utf-8")
    pointer = registry / "active.json"

    publish_family_pointers(registry, {"RF": "abc"}, pointer)

    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["families"]["RF"]["version"] == "abc"
    assert payload["families"]["RF"]["manifest_sha256"]
