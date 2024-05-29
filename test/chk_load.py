import orbax.checkpoint

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

raw_restored = orbax_checkpointer.restore('/tmp/chkpt/myenv_gru/halfcheetah/medium_v2')
print(raw_restored)