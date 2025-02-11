import numpy as np
import torch
from tqdm.auto import tqdm

# Evaluation imports
import evaluation
# Normalization imports
from audio_utils import normalize_audio

import synthesis.utils as utils
import synthesis.io as io
import synthesis.inference as inference

import argparse, os

args = argparse.ArgumentParser()
args.add_argument("--batch_size", type=int, default=1)
args.add_argument("--data_type", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
args.add_argument("--y_filelist", type=str, required=True, help="Path to filelist of test files")
args.add_argument("--tts_ckpt", type=str, required=True, help="Path to MatchaTTS checkpoint")
args.add_argument("--multilingual", default=False, action="store_true", help="Whether current run is multilingual")
args.add_argument("--multi_speaker", default=False, action="store_true", help="Whether current run is multi-speaker")
args.add_argument("--mem_file", type=str, default="mem.pickle", help="Memory record file path")
args.add_argument("--mem_max_entries", type=int, default=100000, help="Maximum number of memory entries")
args.add_argument("--out_dir", type=str, default="synth_output", help="Output directory for synthesis")
args.add_argument("--vocos_config", type=str, default="./configs/vocos/vocos-matcha.yaml", help="Path to Vocos config file")
args.add_argument("--vocos_ckpt", type=str, default="./logs/vocos/multilingual-balanced-dataset/checkpoints/last.ckpt", help="Path to Vocos checkpoint")
args.add_argument("--wandb_name", type=str, default="TTS", help="Name of the WandB run")
args.add_argument("--spk_flag_monolingual", type=str, default="AT", help="Speaker flag for monolingual synthesis")
args = args.parse_args()
BATCHED_SYNTHESIS = args.batch_size != 1

args.data_type = utils.get_dtype(args.data_type)

WANDB_PROJECT = f"TTS"
wandb_name = args.wandb_name + " Batched" if BATCHED_SYNTHESIS else args.wandb_name
wandb_name = wandb_name + f" {args.data_type}"
WANDB_NAME = wandb_name
WANDB_DATASET = "multilingual-test"
WANDB_ARCH = f"MatchaTTS: language embedding, Vocos"

SPK_FLAGS = ["AT", "MJ", "JJ", "NJ"]
SAMPLE_RATE = 22050
## Number of ODE Solver steps
n_timesteps = 10
## Changes to the speaking rate
length_scale = 1.0
## Sampling temperature
temperature = 0.667
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

def synthesis():
    count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"

    model = utils.load_model(args.tts_ckpt, device, args.data_type)
    print(f"Model loaded! Parameter count: {count_params(model)}")
    vocoder = utils.load_vocoder(args.vocos_config, args.vocos_ckpt, device, args.data_type)
    denoiser = None
    index = utils.get_data_index(args.multi_speaker, args.multilingual)
    texts = io.parse_filelist_get_text(args.y_filelist, args.multi_speaker, args.multilingual, sentence_index=index)

    outputs, rtfs = [], []
    rtfs_w = []
    metrics = {}
    throughputs = []
    if BATCHED_SYNTHESIS:
        ckpt = torch.load(args.tts_ckpt, map_location=device)
        hop_length, names, inputs, spks, lang = utils.get_item_batched(ckpt, texts, args.multi_speaker, args.multilingual)
        # compilation runs
        for i in range(5):
            print(f"compile run {i}")
            outputs = inference.batch_synthesis(
                inputs, 
                names, 
                model, 
                vocoder, 
                denoiser, 
                args.batch_size, 
                hop_length, 
                device, 
                SAMPLE_RATE, 
                spks=spks, 
                lang=lang,
                temperature=temperature,
                n_timesteps=n_timesteps,
                length_scale=length_scale
            )
        print(f"synthesis starting")
        outputs = inference.batch_synthesis(
            inputs, 
            names, 
            model, 
            vocoder, 
            denoiser, 
            args.batch_size, 
            hop_length, 
            device, 
            SAMPLE_RATE, 
            spks=spks, 
            lang=lang,
            temperature=temperature,
            n_timesteps=n_timesteps,
            length_scale=length_scale
        )
        
        for i, output in enumerate(outputs):
            normalized_waveforms = []
            rtf_w = output["rtf_w"]
            rtf_w = rtf_w / args.batch_size
            rtf = output["rtf"] / args.batch_size
            throughput = output['throughput']
            for j, wave in enumerate(output['waveform']):
                try:
                    normalized = normalize_audio(wave, sample_rate=SAMPLE_RATE).t()
                except Exception as err:
                    print(f"{output['names'][j]}: {err}")
                    normalized = wave.t()
                normalized_waveforms.append(normalized)
            
            rtfs.append(rtf)
            rtfs_w.append(rtf_w)
            throughputs.append(throughput)

            output['normalized_waveforms'] = normalized_waveforms
            
            io.save_to_folder_batch(output, args.out_dir, SAMPLE_RATE)
            
    else:
        # compilation runs
        for i in range(10):
            print(f"compile run {i}")
            data = texts[i]
            path, spks, lang, text = utils.get_item(data, device)
            dirs = path.split("/")
            name = dirs[len(dirs) - 1].split(".")[0]
            output = inference.synthesise(
                inference.process_text(text, device), 
                model, 
                spks=spks, 
                lang=lang,
                temperature=temperature,
                length_scale=length_scale,
                n_timesteps=n_timesteps
            )
            waveform = inference.to_waveform(output['mel'], denoiser, vocoder)
            output['waveform'] = normalize_audio(waveform, sample_rate=SAMPLE_RATE).t().squeeze()
            
            rtf_w = utils.compute_rtf_w(output, SAMPLE_RATE)
            
        print(f"starting synthesis")
        for i, data in enumerate(tqdm(texts)):
            path, spks, lang, text = utils.get_item(data, device)
            dirs = path.split("/")
            name = dirs[len(dirs) - 1].split(".")[0]
            output = inference.synthesise(
                inference.process_text(text, device), 
                model, 
                spks=spks, 
                lang=lang,
                temperature=temperature,
                length_scale=length_scale,
                n_timesteps=n_timesteps
            )
            waveform = inference.to_waveform(output['mel'], denoiser, vocoder)
            try:
                output['waveform'] = normalize_audio(waveform, sample_rate=SAMPLE_RATE).t().squeeze()
            except Exception as err:
                print(f"{name}: {err}")
                output['waveform'] = waveform.t().squeeze()
                print(f"waveform has NaN? {torch.isnan(waveform).any()}")
            
            rtf_w = utils.compute_rtf_w(output, SAMPLE_RATE)
            rtfs.append(output['rtf'])
            rtfs_w.append(rtf_w)
            
            utils.pretty_print(output, rtf_w, name)
            io.save_to_folder(name, output, args.out_dir, SAMPLE_RATE)
    
    print(f"Experiment: {WANDB_NAME}")
    
    rtfs_mean = np.mean(rtfs)
    rtfs_std = np.std(rtfs)
    rtfs_w_mean = np.mean(rtfs_w)
    rtfs_w_std = np.std(rtfs_w)
    throughput_mean = np.mean(throughputs)
    throughput_std = np.std(throughputs)
    
    metrics["num_ode_steps"] = n_timesteps
    if not BATCHED_SYNTHESIS:
        metrics["rtfs_mean"] = rtfs_mean
        metrics["rtfs_std"] = rtfs_std
        metrics["rtfs_w_mean"] = rtfs_w_mean
        metrics["rtfs_w_std"] = rtfs_w_std
    if BATCHED_SYNTHESIS:
        metrics["throughput_mean"] = throughput_mean
        metrics["throughput_std"] = throughput_std
    
    print(f'"num_ode_steps": {n_timesteps}, "rtfs_mean": {rtfs_mean}, "rtfs_std": {rtfs_std}, "rtfs_w_mean": {rtfs_w_mean}, "rtfs_w_std": {rtfs_w_std}, "throughput_mean": {throughput_mean}, "thoughput_std": {throughput_std}')

    with torch.autocast(device_str, dtype=torch.float32):
        if args.multilingual:
            for spk_flag in SPK_FLAGS:
                stoi, pesq, mcd, f0_rmse, las_rmse, vuv_f1, fd = evaluation.evaluate(args.out_dir, args.y_filelist, spk_flag=spk_flag)
                
                metrics[f"{spk_flag}/stoi"] = stoi
                metrics[f"{spk_flag}/pesq"] = pesq
                metrics[f"{spk_flag}/mcd"] = mcd
                metrics[f"{spk_flag}/f0_rmse"] = f0_rmse
                metrics[f"{spk_flag}/las_rmse"] = las_rmse
                metrics[f"{spk_flag}/vuv_f1"] = vuv_f1
                metrics[f"{spk_flag}/fd"] = fd
                
                print(f'"{spk_flag}/stoi": {stoi}, "{spk_flag}/pesq": {pesq}, "{spk_flag}/mcd": {mcd}, "{spk_flag}/f0_rmse": {f0_rmse}, "{spk_flag}/las_rmse": {las_rmse}, "{spk_flag}/vuv_f1": {vuv_f1}, {spk_flag}/fd": {fd},')
        else:
            stoi, pesq, mcd, f0_rmse, las_rmse, vuv_f1, fd = evaluation.evaluate(args.out_dir, args.y_filelist, args.spk_flag_monolingual)

            metrics[f"{args.spk_flag_monolingual}/stoi"] = stoi
            metrics[f"{args.spk_flag_monolingual}/pesq"] = pesq
            metrics[f"{args.spk_flag_monolingual}/mcd"] = mcd
            metrics[f"{args.spk_flag_monolingual}/f0_rmse"] = f0_rmse
            metrics[f"{args.spk_flag_monolingual}/las_rmse"] = las_rmse
            metrics[f"{args.spk_flag_monolingual}/vuv_f1"] = vuv_f1
            metrics[f"{args.spk_flag_monolingual}/fd"] = fd
            
            print(f'"{args.spk_flag_monolingual}/stoi": {stoi}, "{args.spk_flag_monolingual}/pesq": {pesq}, "{args.spk_flag_monolingual}/mcd": {mcd}, "{args.spk_flag_monolingual}/f0_rmse": {f0_rmse}, "{args.spk_flag_monolingual}/las_rmse": {las_rmse}, "{args.spk_flag_monolingual}/vuv_f1": {vuv_f1}, {args.spk_flag_monolingual}/fd": {fd},')
    
    io.save_python_script_with_data(metrics, WANDB_PROJECT, WANDB_NAME, WANDB_ARCH, WANDB_DATASET, device, filename=args.out_dir + WANDB_NAME.replace(" ", "_") + ".py")
    io.save_metrics(metrics, os.path.join(args.out_dir, "metrics.json"))

if __name__ == "__main__":
    if device_str != "cpu":
        torch.cuda.memory._record_memory_history(max_entries=args.mem_max_entries)
    if args.data_type != None:
        with torch.autocast(device_str, dtype=args.data_type):
            synthesis()
    else:
        synthesis()
    if device_str != "cpu":
        torch.cuda.memory._dump_snapshot(args.mem_file)
        torch.cuda.memory._record_memory_history(enabled=None)
