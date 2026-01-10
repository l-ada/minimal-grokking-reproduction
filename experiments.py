from dataclasses import dataclass, replace, asdict

@dataclass
class TrainConfig:
    optimizer_name: str = "adamw"
    lr: float = 0.0005
    weight_decay: float = 2.0
    rho: float = 0.05
    num_epochs: int = 14000
    batch_size: int = 10000
    modulus: int = 97
    seed: int = 42
    run_name: str = ""
    #tags/comments 
    comments: str = ""
    is_debug: bool = False
    is_overfit: bool = False

    def make_debug(self):
        """Returns a new config instance with debug overrides."""
        return replace(self, 
                       num_epochs=2, 
                       is_debug=True, 
                       run_name=f"DEBUG_{self.run_name}")
    def make_overfit(self):
        """Returns a new config instance with debug overrides."""
        return replace(self, 
                       num_epochs=100, 
                       is_overfit=True, 
                       run_name=f"OVERFIT_{self.run_name}")

    def to_dict(self):
        return asdict(self)
class AdamW(TrainConfig):
    optimizer_name: str = "adamw"
class SAM(TrainConfig):
    optimizer_name: str = "sam"
    rho: float = 0.1
    adaptive=False
class ASAM(TrainConfig):
    optimizer_name: str = "asam"
    rho: float = 0.1
    adaptive=True
class MSAM(TrainConfig):
    optimizer_name: str = "msam"
    betas=(0.9, 0.999)

registry = {"adamw": AdamW,
            "sam": SAM,
            "asam": ASAM,
            "msam": MSAM,}