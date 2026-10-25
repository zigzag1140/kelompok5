import copy, time, glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models import Baseline1DCNN, ProposedCNNBiLSTM
from preprocessing import apply_butterworth_lowpass

CLASSES = ['Walk', 'Sit', 'Stand', 'Fall']

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            loss = criterion(logits, y_b)
            total_loss += loss.item() * x_b.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return {"loss": avg_loss, "acc": acc, "macro_prec": prec * 100, "macro_rec": rec * 100, "macro_f1": f1 * 100}

if __name__ == "__main__":
    print("=== Verifikasi Bobot Model & Evaluasi Sampel Uji ===")
    params = np.load("saved_models/norm_params.npy")
    mean_val, std_val = params[0], params[1]
    use_filter = bool(params[2]) if len(params) > 2 else True

    m_base = Baseline1DCNN()
    m_base.load_state_dict(torch.load("saved_models/baseline_1dcnn.pth", map_location="cpu"))
    m_base.eval()

    m_prop = ProposedCNNBiLSTM()
    m_prop.load_state_dict(torch.load("saved_models/proposed_cnnlstm.pth", map_location="cpu"))
    m_prop.eval()

    for fpath in sorted(glob.glob("demo_samples/*.npy")):
        raw_amp = np.load(fpath)
        proc_amp = apply_butterworth_lowpass(raw_amp, wn=0.18, axis=0) if use_filter else raw_amp
        norm_amp = (proc_amp - mean_val) / (std_val + 1e-8)
        tensor_in = torch.tensor(norm_amp.T, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_base = CLASSES[m_base(tensor_in).argmax(dim=1).item()]
            pred_prop = CLASSES[m_prop(tensor_in).argmax(dim=1).item()]
        print(f"File: {fpath} -> Baseline: {pred_base:<6} | Proposed: {pred_prop:<6}")