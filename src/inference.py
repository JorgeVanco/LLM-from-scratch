from src.model import TransformerLM
from src.optimizers import AdamW
from src.utils import generate_text
from src.tokenizer import Tokenizer
from src.utils import load_checkpoint

if __name__ == "__main__":
    model = TransformerLM(257, 512, 6, 384, 6, 1536, 10000.0)
    optimizer = AdamW(model.parameters())
    load_checkpoint("checkpoints/baseline/checkpoint_3000.pt", model, optimizer)
    
    print(generate_text(model, Tokenizer({i: bytes([i]) for i in range(256)}, [], ["<|endoftext|>"]), "Lily and Max were playing", 512, 100))