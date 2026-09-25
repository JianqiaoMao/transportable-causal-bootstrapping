# Building release artifacts

The package version is declared in `pyproject.toml` and `src/tcb/__init__.py`.
Keep both values consistent. Use a new version for a new published release.

Install the build tools in your development environment:

```sh
python -m pip install build twine
```

Build into an empty staging directory, then check both distributions:

```sh
python -m build --outdir release-staging
python -m twine check release-staging/*.whl release-staging/*.tar.gz
```

The wheel contains the importable package. The standard source distribution
also contains the executed Demo notebooks, their data generator, documentation,
examples, tests, and third-party provenance, as listed in `MANIFEST.in`.
A `-source.zip` archive is a convenience copy of the source distribution for
browsing and running demos; upload the wheel and `.tar.gz` to a Python package
index, rather than this convenience ZIP.

Before replacing the files in `dist`, install the staged wheel with its declared
dependencies in a fresh Python 3.9 or 3.10 environment, run the portable tests
and quickstart, and check the notebook outputs. The bundled separation helper
also ensures sID works with the unmodified published grapl-causal dependency. The optional legacy comparison
tests require the original external experiment scripts and skip if unavailable.

The distributed `SHA256SUMS.txt` records checksums of the wheel, source tarball,
and convenience ZIP. Regenerate checksums whenever any artifact changes.

## Publication metadata

A project license has not yet been selected, and author/contact and repository
URLs have not yet been supplied in the package metadata. Set these deliberately
before public release. `THIRD_PARTY.md` records provenance; it does not grant a
license for this project.

The build and validation commands do not upload or publish anything.
