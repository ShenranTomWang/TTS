# Developing Multilingual Speech Synthesis System for Ojibwe, Mi'kmaq, and Maliseet
### Shenran Wang, Changbing Yang, Mike Parkhill, Chad Quinn, Christopher Hammerly, Jian Zhu
## Getting Started
### Build
Install dependencies listed in [`requirements.txt`](requirements.txt). Alternatively, you can build the environment with [`Dockerfile`](Dockerfile). After that, compile [`monotonic_align`](matcha/utils/monotonic_align/core.c):  
```shell
python "matcha/utils/monotonic_align/setup.py" build_ext --inplace
```

### Inference
#### Data Preparation
Path to your test samples, as well as their speaker/language id and input text should be formatted in each row as follows in a `.txt` file. For our case, AT is 0, MJ is 1, JJ is 2 and NJ is 3. For languages, Maliseet is 0, Mikmaw is 1 and Ojibwe is 2:
```txt
<path to audio>|<speaker id>|<language id>|<input text>   # multilingual, multi-speaker
<path to audio>|<speaker id>|<input text>    # monolingual, multi-speaker
<path to audio>|<language id>|<input text>   # multilingual, mono-speaker
<path to audio>|<input text>        # monolingual, mono-speaker
```
#### Run Inference
To run inference, use [`synthesis.py`](synthesis.py). This will generate a python file that syncs to `WandB`, as most compute clusters may not have internet access. Metric results will also be saved to a `.json` file, and the memory usage snapshot will also be dumped into a `.pickle` file. [`synthesis.py`](synthesis.py) requires path to the TTS model's weights and path to Vocos' weights as args. To view all args, use 
```shell
python synthesis.py --help
```
Depending on how your inference run is configured, you may want to use different args. For example, if you are running monolingual synthesis, the following command should do the basic job:
```shell
python synthesis.py --tts_ckpt <path to tts weights> --y_filelists <path to test filelist> --spk_flag_monolingual <speaker flag> --vocos_ckpt <path to vocos weights> --vocos_config <path to the config .yaml file of vocos>
```
Where `<speaker flag>` is a substring of audio file names that uniquely identifies a speaker, such that we can extract samples of only that speaker from all the multilingual and multi-speaker samples, e.g. JJ, NJ, AT, MJ. Note that this arg is mandatory for monolingual synthesis. If your filelist only contains samples from one speaker, consider passing `""`. 
For multilingual synthesis, add `--multilingual` flag. For multi-speaker synthesis, add `--multi_speaker` flag.

#### Normalizing Audio
After inference, some samples may be too quiet or loud. You can use [`normalize.py`](normalize.py) to normalize audios in a folder:
```shell
python normalize.py --folder <folder of samples to normalize>
```
This will create a `normalized/` folder inside your specified folder that contains all the normalized audios.

### Training
5. To train a model, first go to [`configs/experiment`](configs/experiment) to define your own experiment. Take some of our experiment configs for example, `default` is defined in [`configs/train.yaml`](configs/train.yaml). Then run:
```shell
python matcha/train.py experiment=<your-experiment>
```

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