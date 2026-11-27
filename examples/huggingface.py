import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification

import layertrace as lt

VOCAB, LENGTH, CLS = 24, 16, 1
SPECIAL = [2, 3, 4]


def make_batches(n, seed=0):
    """Dict samples, as a Hugging Face dataset gives them. The label is the most frequent special token."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(5, VOCAB, (n, LENGTH), generator=g)
    for token in SPECIAL:
        counts = torch.randint(0, 4, (n,), generator=g)
        for i in range(n):
            positions = torch.randperm(LENGTH - 1, generator=g)[: counts[i]] + 1
            input_ids[i, positions] = token
    input_ids[:, 0] = CLS
    counts = torch.stack([(input_ids == t).sum(1) for t in SPECIAL], dim=1)
    labels = (counts + 0.1 * torch.arange(len(SPECIAL))).argmax(1)
    mask = torch.ones_like(input_ids)
    return [{"input_ids": input_ids[i], "attention_mask": mask[i], "labels": labels[i]} for i in range(n)]


def main():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=LENGTH,
        num_labels=len(SPECIAL),
    )
    model = BertForSequenceClassification(config)
    train_loader = DataLoader(make_batches(3000, seed=0), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_batches(600, seed=1), batch_size=200)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # dict batches work as they are: layertrace calls model(**batch) without the labels
    # reduce: probe the [CLS] position of each layer, the one the classifier reads
    probes = lt.Probes("runs/huggingface", class_names=[f"token {t}" for t in SPECIAL])
    cls_token = lambda hidden: hidden[:, 0]  # noqa: E731
    for i, layer in enumerate(model.bert.encoder.layer):
        probes.add(layer, lt.LinearProbe(), lt.LayerCKA(every=2), lt.Drift(), name=f"layer{i}", reduce=cls_token)

    for epoch in range(8):
        model.train()
        for batch in train_loader:
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results = probes.run(model, val_loader, epoch)
        print(epoch, {name: round(r["linear_probe_bacc"], 3) for name, r in results.items()})

    print("view it with: layertrace view runs/huggingface")


if __name__ == "__main__":
    main()
