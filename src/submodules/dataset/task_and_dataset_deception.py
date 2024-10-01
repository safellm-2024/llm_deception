import random
from datasets import load_dataset
import torch
from torch import Tensor
import transformer_lens.utils as utils

#
# # Versioin 1
# SYSTEM_PROMPT_HONEST = "You are to answer the following questions in an honest manner."
# SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."
# USER_PROMPT = "Hi AI, is the following statement true or false?"
# SUFFIX = "Oh that's an easy one! The statement is definitely"

# Version 2
SYSTEM_PROMPT_HONEST = "You are to answer the following questions in an honest manner."
SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."
USER_PROMPT = "Is the following statement true or false?"
SUFFIX = "Answer: The statement is"


class ContrastiveDatasetDeception:

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = None
        self.statements = None
        self.answers_positive_correct = None
        self.labels_positive_correct = None
        self.answers_positive_wrong = None
        self.labels_positive_wrong = None

        self.answers_negative_correct = None
        self.labels_negative_correct = None
        self.answers_negative_wrong = None
        self.labels_negative_wrong = None
        self.prompts_negative = None
        self.prompts_positive = None

    def load_and_sample_datasets(self, train_test="train"):
        """
        Load datasets and sample them based on the configuration.

        Returns:
            Tuple of datasets: (dataset_train, dataset_test)
        """
        random.seed(42)
        category = "facts"

        dataset_all = load_dataset("notrichardren/azaria-mitchell-diff-filtered-2")
        dataset = [row for row in dataset_all[f"{category}"]]
        if train_test == "train":
            dataset = random.sample(dataset, self.cfg.n_train)
        else:
            dataset = random.sample(dataset, self.cfg.n_test)
        self.dataset = dataset
        return dataset

    def process_dataset(self):
        statements = [row["claim"] for row in self.dataset]
        labels = [row["label"] for row in self.dataset]
        answers = ["true" if label == 1 else "false" for label in labels]

        answers_positive_wrong = []
        labels_positive_wrong = []
        for label in labels:
            if label == 0:
                label_positive_wrong = 1
                answer_positive_wrong = "true"
            else:
                label_positive_wrong = 0
                answer_positive_wrong = "false"
            answers_positive_wrong.append(answer_positive_wrong)
            labels_positive_wrong.append(label_positive_wrong)

        self.statements = statements
        self.answers_positive_correct = answers
        self.labels_positive_correct = labels
        self.answers_positive_wrong = answers_positive_wrong
        self.labels_positive_wrong = labels_positive_wrong
        return statements, answers, labels

    def construct_contrastive_prompts(self):
        prompts_positive = apply_chat_template(
            self.cfg, self.statements, contrastive_type=self.cfg.contrastive_type[0]
        )
        prompts_negative = apply_chat_template(
            self.cfg, self.statements, contrastive_type=self.cfg.contrastive_type[1]
        )
        (
            answers_negative_correct,
            labels_negative_correct,
            answers_negative_wrong,
            labels_negative_wrong,
        ) = get_negative_answers_and_labels(labels=self.labels_positive_correct)
        self.prompts_positive = prompts_positive
        self.prompts_negative = prompts_negative
        self.answers_negative_correct = answers_negative_correct
        self.labels_negative_correct = labels_negative_correct
        self.answers_negative_wrong = answers_negative_wrong
        self.labels_negative_wrong = labels_negative_wrong
        return (
            prompts_positive,
            prompts_negative,
            answers_negative_correct,
            labels_negative_correct,
        )

    def tokenize_prompts_and_answers(
        self, model, prompts, answers_correct, answers_wrong
    ):

        # tokenization with transformer_lens syntax
        if self.cfg.framework == "transformer_lens":
            prompt_tokens = model.to_tokens(prompts)
            answer_tokens_correct = torch.tensor(
                [
                    model.to_single_token(answers_correct[i])
                    for i in range(len(answers_correct))
                ],
                device=model.cfg.device,
            )
            answer_tokens_wrong = torch.tensor(
                [
                    model.to_single_token(answers_wrong[i])
                    for i in range(len(answers_wrong))
                ],
                device=model.cfg.device,
            )
            prompt_masks = None

        else:
            prompt_inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
            prompt_tokens = prompt_inputs.input_ids.to(model.device)
            prompt_masks = prompt_inputs.attention_mask.to(model.device)

            answer_tokens_correct = torch.tensor(
                [
                    model.to_single_token(answers_correct[i])
                    for i in range(len(answers_correct))
                ]
            ).to(model.device)

            answer_tokens_wrong = torch.tensor(
                [
                    model.to_single_token(answers_wrong[i])
                    for i in range(len(answers_wrong))
                ]
            ).to(model.device)

        print("decoding prompt after tokenization:")
        print(model.tokenizer.decode(prompt_tokens[0]))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        return (prompt_tokens, answer_tokens_correct, answer_tokens_wrong, prompt_masks)


def get_negative_answers_and_labels(labels):
    answers_negative_correct = []
    labels_negative_correct = []
    answers_negative_wrong = []
    labels_negative_wrong = []
    for label in labels:
        if label == 0:
            label_negative_correct = 1
            answer_negative_correct = "true"
            label_negative_wrong = 0
            answer_negative_wrong = "false"
        else:
            label_negative_correct = 0
            answer_negative_correct = "false"
            label_negative_wrong = 1
            answer_negative_wrong = "true"
        answers_negative_correct.append(answer_negative_correct)
        labels_negative_correct.append(label_negative_correct)
        answers_negative_wrong.append(answer_negative_wrong)
        labels_negative_wrong.append(label_negative_wrong)
    return (
        answers_negative_correct,
        labels_negative_correct,
        answers_negative_wrong,
        labels_negative_wrong,
    )


def get_chat_template(model_name: str):
    """
    chat template for different models
    :param model_name:
    :return: chat template for a specific model
    """
    if "gemma" in model_name:
        chat_template = """<start_of_turn>user
        {system_prompt}\n\n{user_prompt}\n\nStatement: {statement}\n\n<end_of_turn>
        <start_of_turn>model
        """
        # suffix = "Answer: The statement is"

    elif "Llama-2" in model_name:
        # template was based on: https://github.com/andyrdt/refusal_direction/blob/main/pipeline/model_utils/llama2_model.py
        chat_template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}\n\nStatement: {statement} [/INST] "
        # suffix = "Answer: The statement is"

    elif "Llama-3" in model_name:
        # template was based on: https://github.com/andyrdt/refusal_direction/blob/main/pipeline/model_utils/llama3_model.py
        chat_template = """"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {user_prompt}\n\nStatement: {statement}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """
        # suffix = "Answer: The statement is"

    elif "Qwen" in model_name:
        chat_template = """"<|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_prompt}\n\nStatement: {statement}<|im_end|>
        <|im_start|>assistant
        """

        # suffix = "Answer: The statement is"

    elif "Yi" in model_name:
        chat_template = """<|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_prompt}\n\nStatement: {statement}<|im_end|>
        <|im_start|>assistant
        """
        # suffix = "Answer: The statement is"

    # for base model
    else:
        chat_template = """
        {system_prompt}\n\n{user_prompt}\n\nStatement: {statement}\n\n
        """
        # suffix = "Answer: The statement is"

    return chat_template, SUFFIX


def apply_chat_template(cfg, statements: list[str], contrastive_type: str = "honest"):
    """
    Construct the full chat prompt by adding chat template for different models.

    :param cfg:
    :param statements:
    :param contrastive_type: either "honest" or "lying"
    :return: the full prompt
    """

    chat_template, suffix = get_chat_template(cfg.model_alias)

    if contrastive_type == "honest":
        prompt = [
            chat_template.format(
                system_prompt=SYSTEM_PROMPT_HONEST,
                user_prompt=USER_PROMPT,
                statement=statement,
            )
            + suffix
            for statement in statements
        ]
    elif contrastive_type == "lying":
        prompt = [
            chat_template.format(
                system_prompt=SYSTEM_PROMPT_LYING,
                user_prompt=USER_PROMPT,
                statement=statement,
            )
            + suffix
            for statement in statements
        ]

    return prompt
