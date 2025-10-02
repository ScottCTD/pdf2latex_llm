from datasets import load_dataset, Dataset
from itertools import islice


stream = load_dataset("OleehyO/latex-formulas-80M", "en",
                      split="train", streaming=True)

def gen():
    yield from islice(stream, 10_000)

small = Dataset.from_generator(gen)   # now it's a regular (map-style) Dataset
small.save_to_disk("latex80m_en_10k")