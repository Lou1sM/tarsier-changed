import os
import yaml
from os.path import join
from tqdm import tqdm
import pandas as pd
import argparse
import json
from natsort import natsorted
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import torch
from tasks.utils import load_model_and_processor
from dataset.utils import format_one_sample

# Load dataset
with open('tvqa-long-annotations_tvqa_val_edited.json') as f:
    full_dset_qs = json.load(f)

show_name_dict = {
    'friends': 'Friends',
    'house': 'House M.D.',
    'met': 'How I Met Your Mother',
    'bbt': 'The Big Bang Theory',
    'castle': 'Castle',
    'grey': "Grey's Anatomy",
}

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    return np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])

def get_texts(split_name, vid_subpath):
    scenes = []
    for fn in natsorted(os.listdir(stexts_rag_caches_dir:=join(ARGS.rag_caches_prefix, 'rag-caches', split_name, vid_subpath, 'scene_texts'))):
        with open(join(stexts_rag_caches_dir, fn)) as f:
            scenes.append(f.read())
    return scenes

def get_showseaseps(show_name_, seas_num_, ep_num_):
    showseaseps = []
    show_names_to_compute = [show_name_] if show_name_ != 'all' else [x for x in natsorted(os.listdir(join(ARGS.rag_caches_prefix, 'rag-caches', 'ours', 'tvqa/'))) if x != 'bbt']

    for show_name in show_names_to_compute:
        seass_to_compute = [seas_num_] if seas_num_ != -1 else natsorted([int(fn[7:]) for fn in os.listdir(join(ARGS.rag_caches_prefix, 'rag-caches', 'ours', 'tvqa', show_name))])
        for seas_num in seass_to_compute:
            if ep_num_ == -1:
                for fn in natsorted(os.listdir(join(ARGS.rag_caches_prefix, 'rag-caches', 'ours', 'tvqa', show_name, f'season_{seas_num}'))):
                    ep_num = int(fn[8:].removesuffix('.mp4'))
                    showseaseps.append((show_name, seas_num, ep_num))
            else:
                showseaseps.append((show_name, seas_num, ep_num_))
    return showseaseps

def answer_qs(show_name, season, episode, model, processor, ep_qs):
    vid_subpath = f'tvqa/{show_name}/season_{season}/episode_{episode}'
    scene_text = '[SCENE_BREAK]'.join(get_texts('ours', vid_subpath))[-ARGS.prompt_prefix:]

    image_dir = f'../amazon_video/data/ffmpeg-keyframes/{vid_subpath}'
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    image_paths = image_paths[::len(image_paths) // ARGS.num_frames][:ARGS.num_frames]
    images = [ToTensor()(np.array(Image.open(fp))).half().to(model.device) for fp in image_paths]

    n_correct = 0
    for qdict in ep_qs['questions']:
        qsent = qdict['q']
        options = '\n'.join(f"{idx}: {qdict[f'a{idx}']}" for idx in range(5))
        prompt = f"""Context from the show:{scene_text}Question: {qsent}Options:{options}Answer with ONLY the number (0-4) of the correct option, nothing else."""

        pred = process_one(model, processor, prompt, image_paths[0], ARGS)
        try:
            ans = int(pred.strip()[0])
            if 0 <= ans <= 4 and ans == qdict['answer_idx']:
                n_correct += 1
        except (ValueError, IndexError):
            pass

    n = len(ep_qs["questions"])
    print(f'VQA accuracy: {n_correct}/{n} = {n_correct/n:.5f}')
    return n_correct, n

def process_one(model, processor, prompt, image_file, args):
    sample = format_one_sample(image_file, prompt)
    batch_data = processor(sample)
    model_inputs = {k: v.to(model.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
    outputs = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    return processor.tokenizer.decode(outputs[0][model_inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default=2)
    parser.add_argument('--ep', type=int, default=-1)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--prompt-prefix', type=int, default=5000)
    parser.add_argument('--rag-caches-prefix', type=str, default='.')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0)
    ARGS = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu'
    data_config = yaml.safe_load(open('configs/tarser2_default_config.yaml', 'r'))
    model, processor = load_model_and_processor("omni-research/Tarsier2-Recap-7b", data_config)
    if ARGS.cpu:
        model.cpu()
    #model.to(device)

    tot_n_correct, tot, all_scores = 0, 0, []
    os.makedirs(out_dir:=f'tvqa-results/tarsier2', exist_ok=True)
    showseaseps = get_showseaseps(ARGS.show_name, ARGS.season, ARGS.ep)

    for show_name, seas, ep in tqdm(showseaseps):
        if f'episode_{ep}' not in full_dset_qs[show_name_dict[show_name]][f'season_{seas}']:
            continue
        ep_qs = full_dset_qs[show_name_dict[show_name]][f'season_{seas}'][f'episode_{ep}']
        new_correct, new_tot = answer_qs(show_name, seas, ep, model, processor, ep_qs)
        tot_n_correct += new_correct
        tot += new_tot
        all_scores.append([show_name, seas, ep, new_correct, new_tot, new_correct/new_tot])

    df = pd.DataFrame(all_scores, columns=['show', 'season', 'episode', 'n_correct', 'n', 'acc'])
    df.to_csv(f'{out_dir}/{ARGS.show_name}_{ARGS.season}-tvqa-results.csv', index=False)

