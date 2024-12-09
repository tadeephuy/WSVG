import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, List, Union
from transformers import BertTokenizer
from health_multimodal.text.utils import CXRBertTokenizer, CXRBertModel, CXR_BERT_COMMIT_TAG, BIOMED_VLP_CXR_BERT_SPECIALIZED

CACHED_CXR_BERT = '/hpcfs/users/a1918064/.cache/huggingface/hub/models--microsoft--BiomedVLP-CXR-BERT-specialized/snapshots/d233f2f4e145bba80c80c60af67349873b81c623'

def get_cxr_bert(checkpoint=None):
    """Load the CXR-BERT model and tokenizer from the `Hugging Face Hub <https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized>`_."""  # noqa: B950
    revision=None
    if checkpoint is None:
        checkpoint=BIOMED_VLP_CXR_BERT_SPECIALIZED
        revision=CXR_BERT_COMMIT_TAG
    tokenizer = CXRBertTokenizer.from_pretrained(checkpoint, revision=revision)
    text_model = CXRBertModel.from_pretrained(checkpoint, revision=revision)
    return tokenizer, text_model

class TextEncoder(nn.Module):
    def __init__(self, tokenizer: BertTokenizer, text_model, max_length=100) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = text_model
        self.max_allowed_input_length = max_length

    def tokenize_input_prompts_(self, prompts: Union[str, List[str]]) -> Any:
        """
        Tokenizes the input sentence(s) and adds special tokens as defined by the tokenizer.
        :param prompts: Either a string containing a single sentence, or a list of strings each containing
            a single sentence. Note that this method will not correctly tokenize multiple sentences if they
            are input as a single string.
        :param verbose: If set to True, will log the sentence after tokenization.
        :return: A 2D tensor containing the tokenized sentences
        """
        prompts = [prompts] if isinstance(prompts, str) else prompts
        self.assert_special_tokens_not_present(" ".join(prompts))

        prompts = [prompt.rstrip("!?.") for prompt in prompts]  # removes punctuation from end of prompt
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=prompts, add_special_tokens=True, padding='longest', return_tensors='pt'
        )

        return tokenizer_output

    def assert_special_tokens_not_present(self, prompt: str) -> None:
        """Check if the input prompts contain special tokens."""
        special_tokens = self.tokenizer.all_special_tokens
        special_tokens.remove(self.tokenizer.mask_token)  # [MASK] is allowed
        if any(map(lambda token: token in prompt, special_tokens)):
            raise ValueError(f"The input \"{prompt}\" contains at least one special token ({special_tokens})")

    def tokenize_input_prompts(self, prompts: Union[str, List[str]]) -> Any:
        tokenizer_output = self.tokenize_input_prompts_(prompts)
        device = next(self.model.parameters()).device
        tokenizer_output.input_ids = tokenizer_output.input_ids.to(device)
        tokenizer_output.attention_mask = tokenizer_output.attention_mask.to(device)

        max_length = tokenizer_output.input_ids.shape[1]
        if tokenizer_output.input_ids.shape[1] > self.max_allowed_input_length:
            raise ValueError(
                f"The sequence length of the input ({max_length}) is "
                f"longer than the maximum allowed sequence length ({self.max_allowed_input_length})."
            )

        return tokenizer_output

    def get_embeddings_from_prompt(
        self, prompts: Union[str, List[str]], normalize: bool = True, verbose: bool = True
    ) -> torch.Tensor:
        """Generate L2-normalised embeddings for a list of input text prompts.

        :param prompts: Input text prompt(s) either in string or list of string format.
        :param normalize: If True, L2-normalise the embeddings.
        :param verbose: If set to True, tokenized words are displayed in the console.
        :return: Tensor of shape (batch_size, embedding_size).
        """

        tokenizer_output = self.tokenize_input_prompts(prompts=prompts)
        txt_emb = self.get_embeddings_from_tokens(tokenizer_output=tokenizer_output, normalize=normalize)

        return txt_emb

    def get_embeddings_from_tokens(self, tokenizer_output, normalize=True):
        txt_emb = self.model.get_projected_text_embeddings(  # type: ignore
            input_ids=tokenizer_output.input_ids.to(self.model.device),
            attention_mask=tokenizer_output.attention_mask.to(self.model.device),
            normalize_embeddings=normalize,
        )
        return txt_emb