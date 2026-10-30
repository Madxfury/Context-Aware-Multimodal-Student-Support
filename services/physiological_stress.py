import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from schemas.observation import PhysiologicalData, StressData
from services.physiological_model import PhysiologicalLSTM

class PhysiologicalStressService:
    """
    A service interface for the V2 BiLSTM physiological stress model.
    Maintains a per-session calibration state using exactly 300 seconds of baseline data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.environ.get("PHYSIOLOGICAL_MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models", "Physiological_V2", "best_bilstm.pth"))
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_available = False
        self.model = None
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        if os.path.exists(model_path):
            try:
                self.model = PhysiologicalLSTM(bidirectional=True)
                # Load weights onto map_location to ensure device compatibility
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.model_available = True
                print(f"Successfully loaded Physiological BiLSTM from {model_path}")
            except Exception as e:
                print(f"Failed to load Physiological BiLSTM: {e}")
        else:
            print(f"Warning: Physiological model not found at {model_path}")

    def _init_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "buffer": [],        # Collects unique 5s sub-windows
                "mu": None,          
                "sigma": None,       
                "calibrated": False, 
                "windows_received": 0
            }

    def predict(self, data: PhysiologicalData) -> StressData:
        """
        Processes a raw 12x14 V2 feature sequence. 
        Calibrates using the first 300 seconds (deduplicated sub-windows).
        Once calibrated, runs inference through the BiLSTM.
        """
        if not self.model_available or data.v2_sequence is None:
            return StressData(stress_level=None, confidence=None, model_available=self.model_available)

        session_id = data.session_id
        self._init_session(session_id)
        session = self.sessions[session_id]

        # data.v2_sequence is expected to be a List[List[float]] of shape [12, 14]
        try:
            seq = np.array(data.v2_sequence, dtype=np.float32)
            if seq.shape != (12, 14):
                print(f"Invalid sequence shape: {seq.shape}. Expected (12, 14)")
                return StressData(stress_level=None, confidence=None, model_available=True)
        except Exception:
            return StressData(stress_level=None, confidence=None, model_available=True)

        if not session["calibrated"]:
            # Deduplicate temporal overlapping steps:
            # First window (windows_received == 0): take all 12 steps.
            # Subsequent overlapping windows: take only the last 6 steps (30 seconds).
            if session["windows_received"] == 0:
                session["buffer"].extend(seq)
            else:
                session["buffer"].extend(seq[6:])
            
            session["windows_received"] += 1

            # 300 seconds of unique 5s sub-windows = 60 sub-windows.
            if len(session["buffer"]) >= 60:
                # Calculate mu and sigma
                calib = np.array(session["buffer"][:60]) # Ensure exactly 60
                mu = np.mean(calib, axis=0)
                sigma = np.std(calib, axis=0)
                sigma[sigma == 0] = 1e-8
                
                session["mu"] = mu
                session["sigma"] = sigma
                session["calibrated"] = True
                
                # Clear raw buffer to save memory
                session["buffer"] = []
                print(f"Session {session_id} successfully calibrated on 300 seconds.")

            # Return unpredicted state while calibrating
            return StressData(stress_level=None, confidence=None, model_available=True)

        # Post-Calibration: Normalize and Infer
        normalized_seq = (seq - session["mu"]) / session["sigma"]
        tensor = torch.tensor(normalized_seq, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, 12, 14]

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
        stress_prob = float(probs[1])
        confidence = float(np.max(probs))
        predicted_class = "Stress" if np.argmax(probs) == 1 else "Baseline"

        return StressData(
            stress_level=stress_prob,
            confidence=confidence,
            predicted_class=predicted_class,
            model_available=True
        )
