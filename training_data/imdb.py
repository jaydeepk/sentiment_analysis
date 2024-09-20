from datasets import load_dataset
from transformers import AutoTokenizer

class IMDB:
    def __init__(self, tokenizer_name="bert-base-uncased", max_length=512, train_size=20000):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.train_size = train_size
        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.is_prepared = False

    def get_data(self):
        if not self.is_prepared:
            self.__load_data()
            self.__preprocess()
            self.is_prepared = True
        return self.train_data, self.val_data, self.test_data

    def __load_data(self):
        try:
            self.dataset = load_dataset("imdb")
            self.train_data = self.dataset['train'].shuffle(seed=42).select(range(self.train_size))
            self.val_data = self.dataset['train'].shuffle(seed=42).select(range(self.train_size, 25000))
            self.test_data = self.dataset['test']
        except Exception as e:
            raise RuntimeError(f"Error loading IMDB dataset: {str(e)}")

    def __preprocess(self):
        try:
            self.train_data = self.train_data.map(self.__tokenize_data, batched=True, batch_size=32)
            self.val_data = self.val_data.map(self.__tokenize_data, batched=True, batch_size=32)
            self.test_data = self.test_data.map(self.__tokenize_data, batched=True, batch_size=32)

            columns_to_return = ['input_ids', 'attention_mask', 'label']
            self.train_data.set_format(type='torch', columns=columns_to_return)
            self.val_data.set_format(type='torch', columns=columns_to_return)
            self.test_data.set_format(type='torch', columns=columns_to_return)
        except Exception as e:
            raise RuntimeError(f"Error preprocessing IMDB dataset: {str(e)}")

    def __tokenize_data(self, examples):
        return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=self.max_length)

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_data_info(self):
           if not self.is_prepared:
               self.get_data()
           return {
               "train_size": len(self.train_data),
               "val_size": len(self.val_data),
               "test_size": len(self.test_data),
               "vocab_size": self.get_vocab_size(),
               "max_length": self.max_length
           }