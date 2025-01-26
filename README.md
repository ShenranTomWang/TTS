# Developing Multilingual Speech Synthesis System for Ojibwe, Mi'kmaq, and Maliseet
### Shenran Wang, Changbing Yang, Mike Parkhill, Chad Quinn, Christopher Hammerly, Jian Zhu
## Getting Started
1. install dependencies listed in [`requirements.txt`](requirements.txt). Alternatively, you can build the environment with [`Dockerfile`](Dockerfile), although it is not recommended as we only used docker environment in very early stages.
2. Compile [`monotonic_align`](matcha/utils/monotonic_align/core.c):  
```shell
python "matcha/utils/monotonic_align/setup.py" build_ext --inplace
```
3. To run inference, use [`synthesis.py`](synthesis.py) (Using environment variables speed up compilation in some compute clusters, compared to using argparse):
```shell
export BATCHED_SYNTHESIS=1      # ignore this if you do not want to use batched processing
export MATCHA_CHECKPOINT="path/to/checkpoint.ckpt"  # this is mandatory
export WANDB_NAME="name of your wandb run"      # this is mandatory
export Y_FILELIST="path/to/filelists.txt"   # this filelist should contain lines of format wav_filepath|speaker_embedding(if any)|language_embedding(if any)|input_text, mandatory
export DATA_TYPE="one of fp16, bf16 and fp32"   # this is mandatory
export SPK_FLAG_MONOLINGUAL="AT"            # Set this only if you are doing monolingual inference
export LANG_EMB=1               # set this to 1 if using multilingual inference, 0 if monolingual
export SPK_EMB=1                # set this to 1 if multi speaker, 0 otherwise
python synthesis.py
```
4. After inference, some samples may be too quiet or loud. You can use [`normalize.py`](normalize.py) to normalize audios in a folder. It is hard-coded so do check it out.
5. To train a model, first go to [`configs/experiment`](configs/experiment) to define your own experiment. Take some of our experiment configs for example, `default` is defined in [`configs/train.yaml`](configs/train.yaml). Then run:
```shell
python matcha/train.py experiment=<your-experiment>
```
6. Have fun!

## Acknowledgement
We thank authors of [Matcha TTS](https://github.com/shivammehta25/Matcha-TTS) for their wonderful codebase

## Citation
```
@inproceedings{
    indigenous-tts,
    title={Developing multilingual speech synthesis system for Ojibwe, Mi{\textquoteright}kmaq, and Maliseet},
    author={Wang, Yang, Parkhill, Quinn, Hammerly and Zhu},
    booktitle={The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL},
    year={2025},
    url={https://openreview.net/forum?id=kPujTsMVQz}
}
```