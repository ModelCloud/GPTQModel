# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import numpy as np
import torch
from datasets import load_dataset, load_from_disk

from gptqmodel.utils.progress import ProgressBar


class Perplexity:
    """
    A class for calculating the perplexity of a language model.
    """

    def __init__(
            self,
            model,
            tokenizer,
            dataset_path="wikitext",
            dataset_name=None,
            split="test",
            text_column="text",
    ):
        """
        Calculate perplexity using the same method as seen in llama.cpp.

        Parameters
        ----------
        model : AutoModelForCausalLM
            The language model for which the perplexity is calculated.
        tokenizer : AutoTokenizer
            The tokenizer corresponding to the model.
        device : str, optional
            The device to run the calculations on. If auto, the device that your model uses
            will be the device used for these calculations. Default is 'auto'.
        dataset_path : str, optional
            The path to the dataset on the Hugging Face dataset hub. Default is 'wikitext'.
        dataset_name : str, optional
            The name of the dataset. Default is None.
        split : str, optional
            The split of the dataset to use. Default is 'test'.
        text_column : str, optional
            The name of the column in the dataset that contains the text data. Default is 'text'.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()

    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"

    def _prepare_data(self):
        """
        Prepares the dataset by loading and formatting.

        Returns
        -------
        str
            The formatted dataset as a single string.
        """
        if self._dataset_path == "wikitext":
            self._dataset_name = "wikitext-2-raw-v1"

        # Load the dataset
        length = 512 if self._dataset_path == "wikitext" else 2048
        if self._dataset_path.startswith("/") or self._dataset_path.startswith("./"):
            if self._dataset_path.endswith(".gz"):
                data = load_dataset(self._dataset_name, data_files=self._dataset_path, split=self._split)
            else:
                data = load_from_disk(self._dataset_path)[self._split]
        else:
            data = load_dataset(self._dataset_path, self._dataset_name, split=self._split)

        datas = []
        for index, sample in enumerate(data):
            text = sample[self._text_column]
            if len(text) >= length:
                # Format the text column of the dataset
                datas.append(" \n" if text == "" else text)
                if len(datas) >= 1024:
                    break

        return "".join(datas)

    @staticmethod
    def softmax(logits):
        """
        Static method for applying the softmax function.

        Parameters
        ----------
        logits : torch.Tensor
            The input to the softmax function.

        Returns
        -------
        np.ndarray
            The output of the softmax function.
        """
        e_x = torch.exp(logits - torch.max(logits))
        return e_x / torch.sum(e_x, dim=0)

    def calculate(self, n_ctx=512, n_batch=512):
        """
        Calculates the perplexity of the language model.

        Parameters
        ----------
        n_ctx : int
            The context size.
        n_batch : int
            The batch size.

        Returns
        -------
        list
            The list of perplexity scores calculated.
        """
        # Tokenize the text
        self._tokenizer.model_max_length = sys.maxsize
        tokens = self._tokenizer(self._text, truncation=False, return_tensors="pt").input_ids.to(self._model.device)

        nll = 0.0  # Negative log likelihood
        count = 0  # Counter for processed tokens
        curr_ppl = 0
        all_perplexity = []

        with ProgressBar(range(len(tokens[0]) // n_ctx), desc="Perplexity: - ") as progress:
            for i in progress:
                # Process each batch of tokens
                nll, count = self._process_batch(i, n_ctx, n_batch, tokens, nll, count)

                # Calculate and display the current perplexity
                curr_ppl = np.exp(nll / count)
                all_perplexity.append(curr_ppl)
                progress.set_description(f"Perplexity: {curr_ppl:.4f}")

        return all_perplexity

    def _process_batch(self, i, n_ctx, n_batch, tokens, nll, count):
        """
        Processes each batch of tokens.

        Parameters
        ----------
        i : int
            The batch index.
        n_ctx : int
            The context size.
        n_batch : int
            The batch size.
        tokens : torch.Tensor
            The tokenized text.
        nll : float
            The current negative log likelihood.
        count : int
            The current count of processed tokens.

        Returns
        -------
        float
            The updated negative log likelihood.
        int
            The updated count of processed tokens.
        """
        start = i * n_ctx
        end = start + n_ctx

        num_batches = (n_ctx + n_batch - 1) // n_batch

        logits = []

        for j in range(num_batches):
            batch_start = start + j * n_batch
            batch_size = min(end - batch_start, n_batch)

            token_org = tokens[0][batch_start].item()

            if j == 0:
                # some models do not set/use bos_token
                if self._tokenizer.bos_token_id is not None:
                    # Replace the first token with the BOS token
                    tokens[0][batch_start] = self._tokenizer.bos_token_id

            # Compute the logits for the current batch of tokens
            batch_logits = self._compute_batch_logits(tokens, batch_start, batch_size)

            tokens[0][batch_start] = token_org

            logits.append(batch_logits)

        # We rely on the fact that attention in the forward pass only looks at previous
        # tokens here, so the logits returned for each token are an accurate representation
        # of what the model would have predicted at that point.
        #
        # Example, we have a context window of 512, we will compute perplexity for each of the
        # last 256 tokens.  Then, we split the input up into context window size chunks to
        # process the entire prompt.

        for j in range(min(512, n_ctx // 2), n_ctx - 1):
            tok_logits = logits[0][0][j]

            # Compute the probability of the next token
            prob = self.softmax(tok_logits)[tokens[0][start + j + 1]]

            # Update the negative log likelihood and the count of processed tokens
            nll += -torch.log(torch.where(prob > 0, prob, torch.tensor(1e-8))).item()
            count += 1

        return nll, count

    def _compute_batch_logits(self, tokens, batch_start, batch_size):
        """
        Computes the logits for a batch of tokens.

        Parameters
        ----------
        tokens : torch.Tensor
            The tokenized text.
        batch_start : int
            The start index of the batch.
        batch_size : int
            The size of the batch.

        Returns
        -------
        torch.Tensor
            The logits for the batch of tokens.
        """
        # Compute the logits without keeping track of gradients
        with torch.no_grad():
            outputs = self._model(tokens[:, batch_start: batch_start + batch_size])
        return outputs.logits.detach()
