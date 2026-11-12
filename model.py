"""Native ALiBi DNABERT2 plus six ordered genomic features.

This architecture is separate from the legacy standard-BERT 40-feature models.
The local DNABERT2 source and pretrained backbone are loaded without downloads.
"""
import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn

FEATURES = {
    "A": ["length_tx_full", "GC_cont_genes", "clust_f100", "pos_exons", "length_cds", "exon_num"],
    "C": ["clust_f100", "long_exon", "GC_cont_101bp", "dist_sj_3_p2000", "length_gene_ex", "length_cds"],
    "G": ["length_tx_full", "exon_stop", "length_cds", "cds", "exon_num", "clust_f100"],
    "T": ["long_exon", "dist_sj_3_p2000", "length_tx_full", "GC_cont_genes", "clust_f100", "UTR3"],
}


def native_classes(directory):
    package = "modcon_local_dnabert2"
    directory = Path(directory).resolve()
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    elif sys.modules[package].__path__ != [str(directory)]:
        raise ValueError("Only one DNABERT2 source directory can be used per process")
    config_cls = importlib.import_module(package + ".configuration_bert").BertConfig
    model_cls = importlib.import_module(package + ".bert_layers").BertModel
    return config_cls, model_cls


class SixFeatureModel(nn.Module):
    def __init__(self, directory, pretrained=False):
        super().__init__()
        config_cls, model_cls = native_classes(directory)
        config = config_cls.from_pretrained(str(directory), local_files_only=True)
        self.dnabert2 = model_cls(config, add_pooling_layer=False)
        self.genomic_proj = nn.Sequential(
            nn.Linear(6, 96), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(96, 128), nn.ReLU(), nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size + 128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2),
        )
        if pretrained:
            original = torch.load(Path(directory) / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
            backbone = {k.removeprefix("bert."): v for k, v in original.items() if k.startswith("bert.")}
            self.dnabert2.load_state_dict(backbone, strict=True)

    def forward(self, input_ids, attention_mask, features):
        hidden = self.dnabert2(input_ids=input_ids, attention_mask=attention_mask)[0]
        # CLS pooling, matching the existing multimodal design. The CLS token
        # attends to the complete, untruncated 800 bp sequence.
        return self.classifier(torch.cat((hidden[:, 0], self.genomic_proj(features)), dim=1))
