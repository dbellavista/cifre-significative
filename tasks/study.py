"""Study scaffolding and registry."""
from invoke import Context, task


@task
def new(c: Context, name: str, title: str, description: str) -> None:
    """Scaffold a study from studies/_template: content folder, code package and a registry test."""
    c.run(f'uv run python scripts/new_study.py {name} "{title}" "{description}"', pty=True)


@task
def list(c: Context) -> None:
    """List the registered studies (name, status, title)."""
    c.run('uv run python -c "from cifresignificative.study import load_studies; [print(s.name, s.status, s.title, sep=\' | \') for s in load_studies().values()]"', pty=True)
