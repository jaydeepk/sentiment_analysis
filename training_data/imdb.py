from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Tuple, Optional


class IMDB:
    def __init__(self, tokenizer_name: str = "bert-base-uncased", max_length: int = 512, train_size: int = 20000) -> None:
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length: int = max_length
        self.train_size: int = train_size
        self.dataset: Optional[Dataset] = None
        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.is_prepared: bool = False

    def get_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        if not self.is_prepared:
            self.__load_data()
            self.__preprocess()
            self.is_prepared = True
        return self.train_data, self.val_data, self.test_data

    def __load_data(self) -> None:
        try:
            self.dataset = load_dataset("imdb")
            self.train_data = self.dataset["train"].shuffle(seed=42).select(range(self.train_size))
            self.val_data = self.dataset["train"].shuffle(seed=42).select(range(self.train_size, 25000))
            self.test_data = self.dataset["test"]
        except Exception as e:
            raise RuntimeError(f"Error loading IMDB dataset: {str(e)}")

    def __preprocess(self) -> None:
        try:
            self.train_data = self.train_data.map(self.__tokenize_data, batched=True, batch_size=32)
            self.val_data = self.val_data.map(self.__tokenize_data, batched=True, batch_size=32)
            self.test_data = self.test_data.map(self.__tokenize_data, batched=True, batch_size=32)

            columns_to_return = ["input_ids", "attention_mask", "label"]
            self.train_data.set_format(type="torch", columns=columns_to_return)
            self.val_data.set_format(type="torch", columns=columns_to_return)
            self.test_data.set_format(type="torch", columns=columns_to_return)
        except Exception as e:
            raise RuntimeError(f"Error preprocessing IMDB dataset: {str(e)}")

    def __tokenize_data(self, examples: dict) -> dict:
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer

    def print_info(self) -> None:
        if not self.is_prepared:
            self.get_data()
        info = {
            "train_size": len(self.train_data),
            "val_size": len(self.val_data),
            "test_size": len(self.test_data),
            "vocab_size": self.get_vocab_size(),
            "max_length": self.max_length,
            "tokenizer": self.tokenizer.name_or_path,
        }
        print("IMDB Dataset Info:")
        for key, value in info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
