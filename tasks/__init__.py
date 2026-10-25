"""Repository glue tasks (``uv run inv ...``): checks, tests, docs freshness.

Run ``uv run inv --list``. Project features stay in ``src/cifresignificative`` and ``scripts``; do not add them here.
Commands use ``rtk`` (condensed output) automatically when it is on the PATH; pass ``--no-rtk`` to disable.
"""
from invoke import Collection

from tasks import dev, docs, test

ns = Collection()
ns.add_collection(Collection.from_module(dev, name="dev"))
ns.add_collection(Collection.from_module(test, name="test"))
ns.add_collection(Collection.from_module(docs, name="docs"))
