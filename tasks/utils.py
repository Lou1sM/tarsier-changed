from models.modeling_tarsier import TarsierForConditionalGeneration, LlavaConfig
# from dataset.processor import Processor
from dataset.tarsier_datamodule import init_processor
import torch
import base64
from tools.color import Color
import yaml

def load_model_and_processor(model_name_or_path, data_config):
    print(Color.red(f"Load model and processor from: {model_name_or_path}"), flush=True)
    if isinstance(data_config, str):
        data_config = yaml.safe_load(open(data_config, 'r'))
    processor = init_processor(model_name_or_path, data_config)
    model_config = LlavaConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = TarsierForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config=model_config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    return model, processor

def file_to_base64(img_path):
    with open(img_path, 'rb') as video_file:
        video_b64_str = base64.b64encode(video_file.read()).decode()
    return video_b64_str

