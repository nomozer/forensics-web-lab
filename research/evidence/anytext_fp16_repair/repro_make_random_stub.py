import sys, torch, pathlib
from transformers import CLIPConfig, CLIPTextModel
root = pathlib.Path(sys.argv[1]); clip = root/"clip-vit-large-patch14"
cfg = CLIPConfig.from_pretrained(str(clip))
m = CLIPTextModel(cfg.text_config); torch.save(m.state_dict(), clip/"pytorch_model.bin"); print("clip stub saved")
from cldm.model import create_model
model = create_model("models_yaml/anytext_sd15.yaml", cond_stage_path=str(clip), use_fp16=False)
sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
torch.save({"state_dict": sd}, root/"anytext_v1.1.ckpt"); print("ckpt stub saved", len(sd), "tensors")
