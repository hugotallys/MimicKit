# Shortcuts for MimicKit + Unitree Go2 (AMP).
# Run from MimicKit/ with the conda env active:  conda activate mimickit
# Override any variable on the command line, e.g.:  make train NUM_ENVS=2048 OUT=output/run2

# ENGINE: newton | isaac_gym | isaac_lab     LOGGER: txt | tb | wandb
PY         ?= python
ENGINE     ?= newton
NUM_ENVS   ?= 4096
LOGGER     ?= tb
OUT        ?= output/go2_amp
MODEL      ?= $(OUT)/model.pt
TEST_ENVS  ?= 1
CLIP       ?= pace
EXTRA      ?=

ENGINE_CFG  = data/engines/$(ENGINE)_engine.yaml
RUN         = $(PY) mimickit/run.py

.DEFAULT_GOAL := help
.PHONY: help check view train train-quick train-view resume test tensorboard

help:
	@echo Targets:
	@echo   make check        - verify torch + Warp see the GPU
	@echo   make view         - play a Go2 reference clip in the viewer (CLIP=walk0..walk3, pace, trot, run)
	@echo   make train        - train AMP on Go2 headless (NUM_ENVS=$(NUM_ENVS), OUT=$(OUT))
	@echo   make train-quick  - ~6 minute smoke test (256 envs, 2M samples, output/smoke_test)
	@echo   make train-view   - train with the viewer on (16 envs, slow, for watching)
	@echo   make resume       - continue training from MODEL=$(MODEL)
	@echo   make test         - run a trained model in the viewer (MODEL=$(MODEL))
	@echo   make tensorboard  - open TensorBoard on OUT (http://localhost:6006)
	@echo Variables: ENGINE=$(ENGINE) NUM_ENVS=$(NUM_ENVS) LOGGER=$(LOGGER) OUT=$(OUT) MODEL=$(MODEL)

check:
	$(PY) -c "import torch, warp as wp; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); wp.init()"

# Reference clip chosen by name: make view CLIP=trot   (EXTRA="--devices cpu" on the Mac)
view:
	$(PY) tools/view_clip.py --clip $(CLIP) --engine_config $(ENGINE_CFG) $(EXTRA)

train:
	$(RUN) --arg_file args/amp_go2_args.txt --engine_config $(ENGINE_CFG) --num_envs $(NUM_ENVS) --visualize false --logger $(LOGGER) --out_dir $(OUT)

train-quick:
	$(RUN) --arg_file args/amp_go2_args.txt --engine_config $(ENGINE_CFG) --num_envs 256 --visualize false --logger txt --max_samples 2000000 --out_dir output/smoke_test

train-view:
	$(RUN) --arg_file args/amp_go2_args.txt --engine_config $(ENGINE_CFG) --num_envs 16 --visualize true --logger $(LOGGER) --out_dir $(OUT)

resume:
	$(RUN) --arg_file args/amp_go2_args.txt --engine_config $(ENGINE_CFG) --num_envs $(NUM_ENVS) --visualize false --logger $(LOGGER) --model_file $(MODEL) --out_dir $(OUT)

test:
	$(RUN) --arg_file args/amp_go2_args.txt --engine_config $(ENGINE_CFG) --mode test --num_envs $(TEST_ENVS) --visualize true --model_file $(MODEL)

tensorboard:
	$(PY) -m tensorboard.main --logdir=$(OUT) --port=6006 --samples_per_plugin scalars=999999
