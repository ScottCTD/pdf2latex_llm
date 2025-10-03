from datasets import load_dataset, Dataset
from itertools import islice


stream = load_dataset("OleehyO/latex-formulas-80M", "en",
                      split="train", streaming=True)

def gen():
    for example in islice(stream, 1_000_000):
        formula = example.get("latex_formula")
        if formula is None or len(formula) <= 256:
            yield example

small = Dataset.from_generator(gen)   # now it's a regular (map-style) Dataset
small.to_parquet("datasets/latex80m_en_1m.parquet")
