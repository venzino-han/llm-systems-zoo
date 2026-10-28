# configs/

YAML configurations for lab scripts, one file per experiment family (`l01_*.yaml`, `l02_*.yaml`, ...).
A script loads a config with `--config configs/<file>.yaml` and CLI flags override individual keys.
The effective configuration is always written to the run directory as `config.json` and hashed into
the run directory name, so two runs with the same config hash are directly comparable.
