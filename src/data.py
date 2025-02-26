import torch
import os
from tqdm import tqdm

def _tokenize(dictionary, path, limit_line=None):
    nb_tokens_in_dictionary = len(dictionary)
    # load document to tokenize
    with open(path, "r", encoding="utf-8") as f:
        document = f.read()

    # Count nb of tokens in text and update the dictionary
    for i, line in enumerate(tqdm(document, desc="Creating dictionary", unit=" lines")):
        if i == limit_line:
            break
        tokens = line.split() + ["<eos>"]
        for token in tokens:
            if token not in dictionary:
                dictionary[token] = nb_tokens_in_dictionary
                nb_tokens_in_dictionary += 1

    # Assign to each token its identifier
    ids = []
    for i, line in enumerate(tqdm(document, desc="Encoding token", unit=" lines")):
        if i == limit_line:
            break
        i += 1
        tokens = line.split() + ["<eos>"]
        for token in tokens:
            ids.append(dictionary[token])
    ids = torch.LongTensor(ids)
    return ids


class Corpus:
    def __init__(self, path=None):
        self._dictionary = {}
        print("Processing train ...")
        self.train = _tokenize(
            dictionary=self._dictionary, path=os.path.join(path, "train.txt")
        )
        print("Processing valid ...")
        self.validation = _tokenize(
            dictionary=self._dictionary, path=os.path.join(path, "validation.txt")
        )
        print("Processing test ...")
        self.test = _tokenize(
            dictionary=self._dictionary, path=os.path.join(path, "test.txt")
        )

    @property
    def vocab_size(self):
        return len(self._dictionary)
    

def batchify(data: torch.Tensor, batch_size):
    # Tính số batch trên data
    num_batches = data.size(0) // batch_size
    # Lấy đủ số lượng batch có thể lấy trên dữ liệu và cắt bỏ những dữ liệu cuối
    data = data.narrow(0, 0, num_batches * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(data, i, stride, evaluation=False):
    seq_len = min(stride, len(data) - 1 - i)
    inputs = data[i : i + seq_len]
    targets = data[i + 1 : i + 1 + seq_len].view(-1)

    if evaluation:
        # Đảm bảo không cần theo dõi gradient
        with torch.no_grad():
            inputs = inputs.clone()
            targets = targets.clone()

    return inputs, targets