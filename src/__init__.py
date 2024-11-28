from src.model import MCNN
from src.dataset import DatasetForDensity


datasets = {
   "mcnn":DatasetForDensity
}

models = {
    "mcnn":MCNN,
}