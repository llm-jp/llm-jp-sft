import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

IGNORE_INDEX: int = -100


@dataclass(frozen=True)
class Example:
    input_ids: list[int]
    labels: list[int]


class CustomDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_files: list[str],
        max_seq_length: int,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.data_files: list[str] = data_files
        self.max_seq_length: int = max_seq_length

        self.num_excluded_examples: int = 0

        self.examples: list[Example] = self._load_examples()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.encode(self.examples[idx])

    def _process_system_user_assistant_message(
        self,
        system_message: str,
        user_message: str,
        assistant_message: str,
    ) -> tuple[list[int], list[int]]:
        prompt: str = f"{system_message}\n\n### 指示:\n{user_message}\n\n### 応答:\n"
        prompt_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_response_input_ids = self.tokenizer.encode(prompt + assistant_message)
        labels: list[int] = [
            IGNORE_INDEX if idx < len(prompt_input_ids) else token_id
            for idx, token_id in enumerate(prompt_response_input_ids)
        ]
        return prompt_response_input_ids, labels

    def _process_user_assistant_message(
        self,
        user_message: str,
        assistant_message: str,
    ) -> tuple[list[int], list[int]]:
        prompt: str = f"\n\n### 指示:\n{user_message}\n\n### 応答:\n"
        prompt_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_response_input_ids = self.tokenizer.encode(prompt + assistant_message)
        labels: list[int] = [
            IGNORE_INDEX if idx < len(prompt_input_ids) else token_id
            for idx, token_id in enumerate(prompt_response_input_ids)
        ]
        return prompt_response_input_ids, labels

    def _load_examples(self):
        examples: list[dict[str, torch.Tensor]] = []
        for data_file in self.data_files:
            with open(data_file, encoding="utf-8") as f:
                loaded_examples = json.load(f)
            for loaded_example in loaded_examples:
                assert len(loaded_example["messages"]) >= 3
                input_ids, labels = self._process_system_user_assistant_message(
                    system_message=loaded_example["messages"][0]["content"],
                    user_message=loaded_example["messages"][1]["content"],
                    assistant_message=loaded_example["messages"][2]["content"],
                )
                total_input_ids: list[int] = input_ids
                total_labels: list[int] = labels
                total_assistant_message: str = loaded_example["messages"][2]["content"]
                if len(loaded_example["messages"]) > 3:
                    num_prompt_response_pairs = (
                        len(loaded_example["messages"]) - 1
                    ) // 2
                    for idx in range(
                        num_prompt_response_pairs - 1
                    ):  # The first pair has already been processed above.
                        input_ids, labels = self._process_user_assistant_message(
                            user_message=loaded_example["messages"][idx * 2 + 3][
                                "content"
                            ],
                            assistant_message=loaded_example["messages"][idx * 2 + 4][
                                "content"
                            ],
                        )
                        total_input_ids.extend(input_ids)
                        total_labels.extend(labels)
                        total_assistant_message += loaded_example["messages"][
                            idx * 2 + 4
                        ]["content"]

                decoded_total_assistant_message_ids: list[int] = []
                for total_input_id, label in zip(total_input_ids, total_labels):
                    if (
                        label != IGNORE_INDEX
                        and total_input_id != self.tokenizer.eos_token_id
                    ):
                        decoded_total_assistant_message_ids.append(total_input_id)
                decoded_total_assistant_message = self.tokenizer.decode(
                    decoded_total_assistant_message_ids
                )
                if (
                    total_assistant_message != decoded_total_assistant_message
                    and total_assistant_message[1:] != decoded_total_assistant_message
                ):
                    # Some original data starts with a half-width space.
                    self.num_excluded_examples += 1
                    continue

                if len(total_input_ids) > self.max_seq_length:
                    self.num_excluded_examples += 1
                    continue

                examples.append(Example(input_ids=total_input_ids, labels=total_labels))
        print(
            f"{self.num_excluded_examples} examples were excluded and {len(examples)} examples were retained."
        )

        for example in examples[:2]:
            print(example)
            print()

        return examples

    @staticmethod
    def encode(example: Example) -> dict[str, list[int]]:
        input_ids: list[int] = example.input_ids
        labels: list[int] = example.labels

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class CustomConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.

        Args:
            dataset (`dataset.Dataset`):
                Dataset with text files.
            seq_length (`int`, *optional*, defaults to `2048`):
                Length of token sequences to return.
            num_of_sequences (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in text buffer.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
    """

    def __init__(
        self,
        dataset,
        seq_length=2048,
        num_of_sequences=1024,
        chars_per_token=3.6,
        shuffle=True,
    ):
        self.dataset = dataset
        self.seq_length = seq_length
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle

    def __len__(self):
        return len(list(self.__iter__()))

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer: list[dict[str, list[int]]] = []
            buffer_len: int = 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    more_examples = False
                    break

            all_token_ids: list[int] = []
            all_labels: list[int] = []
            for example in buffer:
                all_token_ids.extend(example["input_ids"])
                all_labels.extend(example["labels"])

            chunked_examples: list[dict[str, list[int]]] = []
            for i in range(0, len(all_token_ids), self.seq_length):
                chunked_input_ids: list[int] = all_token_ids[i : i + self.seq_length]
                chunked_labels: list[int] = all_labels[i : i + self.seq_length]
                if len(chunked_input_ids) == self.seq_length:
                    chunked_examples.append(
                        {"input_ids": chunked_input_ids, "labels": chunked_labels}
                    )
            if self.shuffle:
                random.shuffle(chunked_examples)
            for chunked_example in chunked_examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(chunked_example["input_ids"]),
                    "labels": torch.LongTensor(chunked_example["labels"]),
                }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "llm-jp/13b-cc-v2-beta-61000step.code20K_en40K_ja60K_ver2.2",
    )

    dataset = CustomDataset(
        tokenizer=tokenizer,
        data_files=[
            "/Users/kodama/project/llm-jp/inst-tuning/tuning/dev/ichikara_004_001_single.json"
        ],
        max_seq_length=2048,
    )
    constant_length_iterator = CustomConstantLengthDataset(
        dataset=dataset, shuffle=False
    )
    for batch in constant_length_iterator:
        print(batch["input_ids"].tolist())
        print(batch["labels"].tolist())
        print()
        break
