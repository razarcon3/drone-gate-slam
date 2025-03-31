import torch



class GateDetector:
    
    def __init__(self):
        # Load the Faster-RCNN Model
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This application requires a CUDA-enabled GPU.")
        self.device = torch.device("cuda")
        self.model = torch.load("full_model.pth", weights_only=False)
        self.model.eval()
        self.model.to(self.device)
        
    