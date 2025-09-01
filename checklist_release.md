# Steps before a release:

## Tests
- Run all tests!
```bash
pytest "tests/"
```

- Check also the jupyter notebook tutorials

## Documentation
- Update whats-new.md
- Bump versions in `pyproject.toml` and `__init__.py`

## Tag commit
```bash
git tag -a v0.1.0 <commit-hash> -m "Initial release version 0.1.0"
git push origin --tags
```

## Release
```bash
python3 -m build
twine upload dist/sunscanpy-1.0.0-py3-none-any.whl dist/sunscanpy-1.0.0.tar.gz
```
For a test upload to testPyPI:
```bash
twine upload -r testpypi dist/*
```