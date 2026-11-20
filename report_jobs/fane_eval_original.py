"""Original Swin-FANE checkpoint (Models/Swin_FANE_Best_Model.pth) and HSEmotion enet_b2_7 on the
406 unseen FANE test images; the original model also on the whole 4,616-file project test split.
Saves per-class probabilities and Grad-CAM overlays (app.py's SwinGradCAM, same target layer and blend)."""
import os, sys, json, time, math
import numpy as np, pandas as pd, torch, cv2
from PIL import Image
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from fane_common import *

HERE = os.path.dirname(os.path.abspath(__file__)); PROJECT = os.path.dirname(HERE); OUT = os.path.join(HERE, 'outputs')
os.makedirs(OUT, exist_ok=True)
root = fane_root(); dev = device(); seed_everything()
split = pd.read_csv(os.path.join(HERE, 'fane_split_report.csv'))
test = split[split.split == 'test'].reset_index(drop=True)
status = {'device': str(dev), 'torch': torch.__version__, 'fane_root': root}

# --- HSEmotion (preprocessing copied from app.py expression_probabilities) ---
import onnxruntime
onnx_path = os.path.join(PROJECT, 'Models', 'hsemotion', 'enet_b2_7.onnx')
sess = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
def hse(rgb):
    x = cv2.resize(rgb, (260, 260)).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    logits = sess.run(None, {'input': x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)})[0][0]
    e = np.exp(logits - logits.max()); return e / e.sum()
P = np.stack([hse(np.asarray(Image.open(os.path.join(root, f)).convert('RGB'))) for f in test.file])
probs_frame(test, P, ['file_in_split']).to_csv(os.path.join(OUT, 'hsemotion_test406_probs.csv'), index=False)
status['hsemotion'] = 'done'

# --- Original Swin-FANE ---
ckpt = os.path.realpath(os.path.join(PROJECT, 'Models', 'Swin_FANE_Best_Model.pth'))
status['original_checkpoint'] = ckpt
if not os.path.isfile(ckpt):
    status['original'] = 'checkpoint not found'
    json.dump(status, open(os.path.join(OUT, 'fane_eval_original_status.json'), 'w'), indent=2); sys.exit(0)
model = CustomSwinTransformer(pretrained=False)
state = torch.load(ckpt, map_location='cpu', weights_only=True)
if isinstance(state, dict) and 'model_state_dict' in state:
    state = state['model_state_dict']
model.load_state_dict(state, strict=True); model.to(dev).eval()

t0 = time.time()
P = predict(model, DataLoader(CsvImages(test, root), batch_size=32), dev)
probs_frame(test, P, ['file_in_split']).to_csv(os.path.join(OUT, 'orig_swin_test406_probs.csv'), index=False)
sp = pd.read_csv(os.path.join(root, 'split_used_in_project.csv'))
full = sp[sp.split == 'test'].rename(columns={'same_image_in_images_folder': 'file'}).reset_index(drop=True)
P = predict(model, DataLoader(CsvImages(full, root), batch_size=32), dev)
probs_frame(full, P, ['file_in_split', 'also_in_train']).to_csv(os.path.join(OUT, 'orig_swin_test4616_probs.csv'), index=False)
status['original'] = f'done in {time.time()-t0:.0f} s'

# --- Grad-CAM: app.py SwinGradCAM logic, run on CPU ---
class SwinGradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'acts', o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grads', go[0].detach()))
    def generate(self, x):
        out = self.model(x); k = int(out.argmax(1)); out[0, k].backward()
        a, g = self.acts, self.grads
        if a.ndim == 3:
            cam = ((g.mean(dim=1, keepdim=True)) * a).sum(2).squeeze(0); s = int(math.sqrt(cam.shape[0])); cam = cam.reshape(s, s)
        elif a.shape[-1] == self.model.backbone.num_features:
            cam = (g.mean(dim=(1, 2), keepdim=True) * a).sum(-1).squeeze(0)
        else:
            cam = (g.mean(dim=(2, 3), keepdim=True) * a).sum(1).squeeze(0)
        cam = np.maximum(cam.cpu().numpy(), 0); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, k, float(torch.softmax(out, 1)[0, k])
model.to('cpu'); cammer = SwinGradCAM(model, model.backbone.layers[-1].blocks[-1].norm2)
pr = pd.read_csv(os.path.join(OUT, 'orig_swin_test406_probs.csv'))
picks = []
for e in EMOTIONS:
    g = pr[(pr.emotion == e) & (pr.predicted == e)].sort_values('file')
    picks.append((g.iloc[0] if len(g) else pr[pr.emotion == e].sort_values('file').iloc[0]))
meta = []
os.makedirs(os.path.join(OUT, 'gradcam'), exist_ok=True)
for r in picks:
    img = Image.open(os.path.join(root, r.file)).convert('RGB')
    cam, k, conf = cammer.generate(TRANSFORM(img).unsqueeze(0)); model.zero_grad(set_to_none=True)
    face = cv2.cvtColor(np.asarray(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (224, 224))), cv2.COLORMAP_JET)
    name = os.path.basename(r.file).replace('.jpg', '')
    cv2.imwrite(os.path.join(OUT, 'gradcam', f'{name}_face.png'), face)
    cv2.imwrite(os.path.join(OUT, 'gradcam', f'{name}_cam.png'), cv2.addWeighted(face, 0.45, heat, 0.55, 0))
    meta.append({'file': r.file, 'true': r.emotion, 'predicted': EMOTIONS[k], 'confidence': conf})
json.dump(meta, open(os.path.join(OUT, 'gradcam', 'gradcam_examples.json'), 'w'), indent=2)
status['gradcam'] = 'done'
json.dump(status, open(os.path.join(OUT, 'fane_eval_original_status.json'), 'w'), indent=2)
print(json.dumps(status, indent=2))
