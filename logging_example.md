### Setup

```bash
pip install durbango
cd ~/transformers # wherever your fork is, I'm assuming upstream is the real repo
git fetch upstream
git checkout mem-prof-bart

```
Base command:
```bash
pytest --tb=short tests/test_bart_memory.py -s
```
This generates logs like `hf_short_generate.txt`. using the `save_logs` functionality
```
start: GPU: 6.761GB CPU: 5.846GB
done encoder, outputs shaped torch.Size([1024, 24, 1024]): GPU: 7.935GB CPU: 5.846GB
done: GPU: 7.974GB CPU: 5.848GB
```

### Suggested Workflow


Get git setup to checkout PRs by number
my `.git/config` (for my fork) looks like this:
```
[remote "upstream"]
    url = git@github.com:huggingface/transformers.git
    fetch = +refs/heads/*:refs/remotes/upstream/*
    fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*
```
([inspired by this](https://gist.github.com/piscisaureus/3342247))
Then checkout PRs and the testing script and run:
```bash
git checkout pr/3370 
git checkout mem-prof-bart tests/test_bart_memory.py small_test.source
pytest --tb=short tests/test_bart_memory.py -s -k hf_
```
If you want to force CUDA to empty GPU RAM, you can run tests 1 by 1:
like for forward 
```bash
pytest --tb=short tests/test_bart_memory.py -s -k hf_fwd
```
or for generate
```
pytest --tb=short tests/test_bart_memory.py -s -k hf_short_gen
```

### Add more logging statements
in any child of `BartForConditionalGeneration`, like `PretrainedModel`, you can add`self.log_mem(optional_message)` and your message, a timestamp, and the current gpu mem will show up 
in a csv if you call `model.save_logs_to_csv()`. There is one example logging statement in `modeling_utils.py` on the `mem-prof-bart` branch. 

### Benchmarking fairseq
If you clone/fork fairseq and run `pip install -e .` you can start adding `log_mem` statements inside fairseq too, and the fairseq tests should run.


### How it works
there are two ways to use:
1) make all your modules inherit from `durbango.logging_utils.LoggingMixin` -- annoying because you have to merge changes between branches.
2) use `durbango.logging_patch.patch_module_with_memory_mixin`,as the test script does, which adds the required methods at runtime.

The underlying data collection is py3nvml, same as `start_memory_tracing`. Then there is some very simple pandas.
Code in this repo in `logging_utils.py`

