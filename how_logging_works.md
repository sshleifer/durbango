### For transformers/bart example
```bash
pip install durbango
git fetch upstream
git checkout mem-prof-bart

```
Base command:
```bash
pytest --tb=short tests/test_bart_memory.py -s
```

This generates logs like `hf_short_generate.txt`.
```

``


### Add more logging statements
in code `self.log_mem`



### How it works
