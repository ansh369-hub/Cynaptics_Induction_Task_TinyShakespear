#dataloader.py
from datasets import load_dataset

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca_prompt(example: dict) -> dict:
    """ format a single alpaca example into the prompt template."""
    if example.get("input") and example["input"].strip():
        text = PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        text = PROMPT_WITHOUT_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text}


def load_alpaca_dataset(split: str = "train", test_size: float = 0.1, seed: int = 42):
    """
    load the alpaca dataset from HF and apply the prompt template.
    """
    dataset = load_dataset("tatsu-lab/alpaca", split="train")


    dataset = dataset.map(format_alpaca_prompt)

    if split == "all":
        return dataset


    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    if split == "train":
        return split_dataset["train"]
    elif split == "test":
        return split_dataset["test"]
    else:
        return split_dataset


if __name__ == "__main__":
    # load and print a few samples
    train_data = load_alpaca_dataset(split="train")
    test_data = load_alpaca_dataset(split="test")

    print(f"\ntrain size: {len(train_data)}")
    print(f"test size:  {len(test_data)}")
    print(f"\n{'='*60}")
    print("formatted prompt:\n")
    print(train_data[0]["text"])


