import sys, json, traceback, importlib.util, pathlib
import numpy as np
from PIL import Image, ImageDraw
REPO = pathlib.Path(r"D:/Documents/forensics-web-lab")
spec = importlib.util.spec_from_file_location("runner", REPO/"scripts/research/run_anytext_colab.py"); runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
sys.path.insert(0, ".")
from ml.evaluation.anytext_document_adapter import prepare_anytext_editing_inputs
plan = json.load(open(REPO/"research/evidence/document_pilot/anytext_feasibility_plan.json", encoding="utf-8"))
att = plan["attempts"][0]
img = Image.new("RGB", (576, 864), (235, 230, 220)); d = ImageDraw.Draw(img)
x0,y0,x1,y1 = att["target_bbox"]; d.rectangle([x0,y0,x1,y1], outline=(0,0,0)); d.text((x0+2,y0+2), "280,000", fill=(0,0,0))
prep = prepare_anytext_editing_inputs(image=img, target_bbox=att["target_bbox"], canvas_size=512, context_margin_ratio=0.5)
from modelscope.pipelines import pipeline
pipe = pipeline("my-anytext-task", model=sys.argv[1], use_fp16=True, use_translator=False, font_path="C:/Windows/Fonts/arial.ttf")
print("PIPELINE_INIT_OK")
try:
    out = runner.run_anytext_inference(pipe=pipe, ori_image_canvas=prep["ori_image_canvas"], draw_pos_canvas=prep["draw_pos_canvas"], prompt=att["prompt"], target_text=att["target_text"], seed=att["seed"], sampling_params=att["sampling_parameters"], a_prompt=att["a_prompt"], n_prompt=att["n_prompt"])
    print("INFERENCE_OK", out[1], out[2])
except Exception:
    traceback.print_exc(); sys.exit(3)
