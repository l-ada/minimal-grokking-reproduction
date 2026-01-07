
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
    # model
    # dataset/dataloader
    #tags/comments 
    comments: str = ""