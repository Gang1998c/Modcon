#!/usr/bin/env python3
"""ModCon scoring with native DNABERT2 + six base-specific genomic features.

GitHub-compatible CLI and four-column output. CSV/TSV and the earlier local CLI
are also supported. All files in this folder can be copied out of this project.
"""
import argparse
import collections
import contextlib
import csv
import hashlib
import json
import logging
import mmap
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
from liftover import ChainFile
from transformers import AutoTokenizer
from model import FEATURES, SixFeatureModel

GF_NAMES = ["UTR5","UTR3","cds","Stop_codons","Start_codons","TSS","TSS_A",
 "exon_stop","alternative_exon","constitutive_exon","internal_exon","long_exon",
 "last_exon","last_exon_400bp","last_exon_sc400","intron","pos_UTR5","pos_UTR3",
 "pos_cds","pos_exons","dist_sj_5_p2000","dist_sj_3_p2000","length_UTR3","length_UTR5",
 "length_cds","length_gene_ex","length_gene_full","length_tx_exon","length_tx_full",
 "clust_f1000","clust_f100","dist_nearest_p2000","dist_nearest_p200","sncRNA","lncRNA",
 "isoform_num","exon_num","GC_cont_genes","GC_cont_101bp","GC_cont_101bp_abs"]
COMP = str.maketrans("ACGTRYMKBDHVNSW", "TGCAYRKMVHDBNSW")
logger = logging.getLogger("modcon_six")


def reverse_complement(seq):
    return seq.translate(COMP)[::-1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def sep(path):
    return "\t" if Path(path).suffix.lower() == ".tsv" else ","


def read_table(path, **kwargs):
    return pd.read_csv(path, sep=sep(path), encoding="utf-8-sig", **kwargs)


def save_table(frame, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, sep=sep(path), index=False, lineterminator="\n")
    os.replace(temp, path)


class Fasta:
    """Read-only, indexed FASTA access; never creates files beside a genome."""
    def __init__(self, path):
        self.index = {}
        with open(str(path) + ".fai", encoding="utf-8") as f:
            for line in f:
                name, length, offset, bases, byte_width, *_ = line.strip().split("\t")
                self.index[name] = tuple(map(int, (length, offset, bases, byte_width)))
        self.handle = open(path, "rb")
        self.data = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    def fetch(self, chrom, start, end, strand="+"):
        length, offset, bases, byte_width = self.index[chrom]
        if not 0 <= start < end <= length or strand not in ("+", "-"):
            raise ValueError(f"Invalid FASTA interval: {chrom}:{start}-{end} {strand}")
        begin = offset + start // bases * byte_width + start % bases
        last = end - 1
        stop = offset + last // bases * byte_width + last % bases + 1
        sequence = self.data[begin:stop].replace(b"\n", b"").replace(b"\r", b"").decode("ascii").upper()
        if len(sequence) != end - start:
            raise ValueError("FASTA index disagrees with line layout")
        return reverse_complement(sequence) if strand == "-" else sequence

    def close(self):
        self.data.close()
        self.handle.close()


def aliases_for(fasta, report=None):
    aliases = {name: name for name in fasta.index}
    if report:
        for line in report.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10 or fields[9] not in fasta.index:
                continue
            for name in (fields[0], fields[4], fields[6], fields[9]):
                if name != "na":
                    if name in aliases and aliases[name] != fields[9]:
                        raise ValueError(f"Ambiguous chromosome alias: {name}")
                    aliases[name] = fields[9]
    for chrom in list(map(str, range(1, 23))) + ["X", "Y"]:
        if "chr" + chrom in fasta.index:
            aliases[chrom] = "chr" + chrom
    if "chrM" in fasta.index:
        aliases.update({"MT": "chrM", "M": "chrM"})
    return aliases


def model_path(directory, base):
    # Published directory is U, while checkpoint metadata and DNA alphabet use T.
    folders = ["model_U", "model_T"] if base == "T" else ["model_" + base]
    for folder in folders:
        path = directory / folder / "best_model.pth"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing six-feature model for {base}: {directory}; expected {folders}/best_model.pth")


def normalize_sites(path, fasta, aliases, invalid, work):
    with path.open(encoding="utf-8-sig", newline="") as f:
        raw_header = next(csv.reader(f, delimiter=sep(path)), [])
    if len(raw_header) != len(set(raw_header)):
        raise ValueError("Input contains duplicate column names")
    original = read_table(path, dtype=str, keep_default_na=False)
    if original.empty or original.columns.duplicated().any():
        raise ValueError("Input must have a unique header and at least one row")
    header = {str(c).strip().lower(): c for c in original.columns}
    if len(header) != len(original.columns):
        raise ValueError("Ambiguous input header after case/space normalization")
    chr_col = header.get("seqnames", header.get("contig"))
    pos_col = header.get("position", header.get("gen_position"))
    if not chr_col or not pos_col or "strand" not in header:
        raise ValueError("Require seqnames/position/strand (1-based), or contig/gen_position/strand (0-based)")
    position = pd.to_numeric(original[pos_col], errors="coerce")
    if pos_col == header.get("gen_position"):
        position = position + 1
    sites, dropped = [], []
    for i, values in enumerate(original.itertuples(index=False, name=None)):
        row = dict(zip(original.columns, values))
        q, strand = position.iloc[i], row[header["strand"]].strip()
        chrom = aliases.get(row[chr_col].strip(), row[chr_col].strip())
        reason = None
        if not np.isfinite(q) or q != int(q) or q < 1 or strand not in ("+", "-"):
            reason = "invalid_coordinate_or_strand"
        elif chrom not in fasta.index:
            reason = "missing_hg38_reference"
        else:
            q = int(q)
            for name in ["position", "position_1based", "gen_position"]:
                if name in header:
                    raw = pd.to_numeric(row[header[name]], errors="coerce")
                    if raw != q - (name == "gen_position"):
                        raise ValueError(f"Coordinate columns disagree at input row {i+1}")
            # Strand-oriented 400 upstream + target + 399 downstream.
            start = q - (401 if strand == "+" else 400)
            if start < 0 or start + 800 > fasta.index[chrom][0]:
                reason = "hg38_window_out_of_bounds"
        if reason:
            dropped.append({"source_row_1based": i+1, "reason": reason})
            continue
        base = fasta.fetch(chrom, q-1, q, strand)
        if "ref_base" in header and row[header["ref_base"]].strip().upper() not in ("", base):
            raise ValueError(f"ref_base disagrees with exact strand-oriented hg38 site at input row {i+1}")
        sites.append((i+1, chrom, q, strand, base))
    save_table(pd.DataFrame(dropped, columns=["source_row_1based", "reason"]), work / "dropped_sites.csv")
    if dropped and invalid == "error":
        raise ValueError(f"{len(dropped)} invalid sites; inspect dropped_sites.csv or use --invalid-sites drop")
    if not sites:
        raise ValueError("No scoreable hg38 windows remain")
    result = pd.DataFrame(sites, columns=["row_id", "seqnames", "position", "strand", "ref_base"])
    return original.iloc[result.row_id.to_numpy()-1].reset_index(drop=True), result


def sequence_800(fasta, row):
    start = int(row.position) - (401 if row.strand == "+" else 400)
    seq = fasta.fetch(row.seqnames, start, start+800, row.strand)
    if len(seq) != 800 or seq[400] != row.ref_base:
        raise ValueError("Incorrect target placement in model window")
    if set(seq) - set("ACGTN"):
        raise ValueError("Model input contains unsupported DNA ambiguity characters beyond ACGTN")
    return seq


def alignment_score(seq1, seq2):
    if len(seq1) != 11 or len(seq2) != 11 or seq1[5] != seq2[5] or seq1[5] not in "ACGT":
        return 0.0
    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    score = sum(2 if a == b and a in "ACGT" else 1 if (a, b) in transitions else 0 for a, b in zip(seq1, seq2))
    return round(score / 22.0, 2)


def compare_species(row, human, target, converter, known):
    mapped = converter.query(row.seqnames, int(row.position)-1) or []
    result = dict(score=0.0, status="-", alignment_reason="unmapped", target_seqnames="",
                  target_position="", target_strand="", mapping_count=len(mapped))
    if not mapped:
        return result
    chrom, t0, orientation = mapped[0]
    if orientation not in ("+", "-"):
        raise ValueError("Unsupported chain orientation")
    ts = row.strand if orientation == "+" else ("-" if row.strand == "+" else "+")
    result.update(target_seqnames=chrom, target_position=int(t0)+1, target_strand=ts)
    if chrom not in target.index:
        result["alignment_reason"] = "missing_target_reference"
    elif t0 < 5 or t0+6 > target.index[chrom][0]:
        result["alignment_reason"] = "target_window_out_of_bounds"
    else:
        hs = human.fetch(row.seqnames, int(row.position)-6, int(row.position)+5, row.strand)
        tt = target.fetch(chrom, int(t0)-5, int(t0)+6, ts)
        if hs[5] != tt[5]:
            result["alignment_reason"] = "center_mismatch"
        elif hs[5] not in "ACGT":
            result["alignment_reason"] = "unsupported_center_base"
        else:
            result.update(score=alignment_score(hs, tt), alignment_reason="ok",
                          status="conserved" if f"{chrom}:{int(t0)+1} {ts}" in known else "-")
    return result


def reference_scores(args, sites, aliases):
    """Stream the large reference, retaining only requested coordinate keys."""
    wanted = set(zip(sites.seqnames, sites.position, sites.strand))
    header = {str(c).strip().lower(): c for c in read_table(args.ontdata_path, nrows=0).columns}
    required = ["seqnames", "start", "strand", "rank_noes", "pm_score"]
    if not set(required).issubset(header):
        raise ValueError("ontdata needs seqnames,start,strand,rank_noes,pm_score")
    found = {}
    for chunk in read_table(args.ontdata_path, usecols=[header[n] for n in required], chunksize=200000):
        chunk.columns = chunk.columns.str.strip().str.lower()
        for c, q, s, n, p in chunk[["seqnames","start","strand","rank_noes","pm_score"]].itertuples(index=False, name=None):
            key = (aliases.get(str(c), str(c)), int(q), str(s))
            if key not in wanted:
                continue
            values = tuple(float(v) if pd.notna(v) and np.isfinite(float(v)) else 0.0 for v in pd.to_numeric(pd.Series([n, p]), errors="coerce"))
            if key in found and values != found[key]:
                raise ValueError(f"Conflicting noes/pm reference rows for {key}")
            found[key] = values
    return pd.DataFrame([(found.get(key, (0.0, 0.0)) + (key in found,)) for key in zip(sites.seqnames, sites.position, sites.strand)],
                        columns=["noes", "pm", "ontdata_matched"])


def cross_species(args, sites, human):
    columns = {}
    for species, fasta_path, chain_path, catalog, col in [
        ("mm39", args.mm39_fa, args.mm39_chain, args.mouse_raw_path, "site_pos_mm39"),
        ("sus11", args.sus11_fa, args.sus11_chain, args.pig_raw_path, "site_pos_sus11")]:
        logger.info("Cross-species comparison: %s", species)
        converter = ChainFile(str(chain_path), "hg38", "mm39" if species == "mm39" else "susScr11")
        known = set(read_table(catalog, usecols=[col])[col].dropna().astype(str).str.strip())
        target = Fasta(fasta_path)
        try:
            values = [compare_species(row, human, target, converter, known) for row in sites.itertuples()]
        finally:
            target.close()
        for key in values[0]:
            name = f"seq_score_{species}" if key == "score" else f"{key}_{species}"
            columns[name] = [v[key] for v in values]
    return pd.DataFrame(columns)


def resource_identity(args):
    # Genome/reference catalogs use size+mtime, while model/code/chain/index
    # contents are hashed. A changed resource invalidates the entire work cache.
    identity = {"pipeline": "six-feature-v1", "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    identity["resources"] = {}
    big = [args.hg38_fa, args.mm39_fa, args.sus11_fa, args.ontdata_path, args.mouse_raw_path, args.pig_raw_path]
    small = [args.user_df_path, args.r_script_path, args.hg19_chain, args.mm39_chain, args.sus11_chain,
             HERE / "master_pipeline.py", HERE / "model.py"]
    small += [Path(str(p)+".fai") for p in big[:3]]
    small += [args.dnabert_model_path / name for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "configuration_bert.py", "bert_layers.py", "bert_padding.py", "flash_attn_triton.py"]]
    small += [model_path(args.base_model_dir, base) for base in "ACGT"]
    if args.assembly_report:
        small.append(args.assembly_report)
    for path in big + small:
        stat = path.stat()
        entry = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path in small:
            entry["sha256"] = sha256(path)
        identity["resources"][str(path)] = entry
    return identity


def cached(args, stage, files):
    path = args.work / f"{stage}_cache.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    for name in files:
        file = args.work / name
        if not file.is_file() or sha256(file) != record.get(name):
            raise ValueError(f"Corrupt cached {stage}: {name}; use a fresh --work directory")
    logger.info("Reuse verified %s cache", stage)
    return True


def mark_cached(args, stage, files):
    save_json(args.work / f"{stage}_cache.json", {name: sha256(args.work/name) for name in files})


def generate_features(args, sites):
    names = ["genomic_features.csv"] + [f"model_features_{base}.csv" for base in "ACGU"]
    if not cached(args, "features", names):
        save_table(sites, args.work / "sites.csv")
        command = ["Rscript", "--vanilla", str(args.r_script_path), str(args.work/"sites.csv"),
                   str(args.work/"genomic_features.csv"), str(args.hg19_chain)]
        logger.info("Generate GF on all %s retained sites using Rscript", len(sites))
        with (args.work / "feature_generation.log").open("w", encoding="utf-8") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
        mark_cached(args, "features", names)
    result = read_table(args.work / "genomic_features.csv", keep_default_na=False)
    if result.row_id.tolist() != sites.row_id.tolist() or not np.isfinite(result[GF_NAMES].to_numpy(float)).all():
        raise ValueError("R features lost row alignment or contain nonfinite values")
    for base in "ACGT":
        selected = read_table(args.work / f"model_features_{'U' if base=='T' else base}.csv")
        if selected.columns.tolist() != ["row_id"] + FEATURES[base]:
            raise ValueError(f"Wrong six-feature order in R output: {base}")
        mask = sites.ref_base == base
        if selected.row_id.tolist() != sites.loc[mask, "row_id"].tolist():
            raise ValueError("R model-specific rows changed order")
        np.testing.assert_allclose(selected[FEATURES[base]].to_numpy(float), result.loc[mask, FEATURES[base]].to_numpy(float), rtol=1e-12, atol=1e-12)
    return result


def load_checkpoint(args, base, device):
    path = model_path(args.base_model_dir, base)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state.get("base") != base or state.get("feature_names") != FEATURES[base]:
        raise ValueError(f"Checkpoint base/feature order mismatch: {path}")
    scaler = state["preprocessing"]
    if scaler["feature_names"] != FEATURES[base] or scaler["sequence_bp"] != 800:
        raise ValueError("Checkpoint preprocessing incompatible")
    mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    if mean.shape != (6,) or scale.shape != (6,) or not np.isfinite([mean, scale]).all() or (scale <= 0).any():
        raise ValueError("Invalid saved six-feature scaler")
    model = SixFeatureModel(args.dnabert_model_path)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval(), scaler, path


def predict(args, sites, gf, fasta):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise ValueError("Requested CUDA is unavailable")
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("--precision bf16 requires a BF16-capable CUDA GPU")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(args.dnabert_model_path), local_files_only=True, clean_up_tokenization_spaces=False)
    probabilities = np.zeros(len(sites), float)
    classes = np.full(len(sites), -1, dtype=int)
    diagnostics = {}
    for base in "ACGT":
        indices = np.flatnonzero(sites.ref_base.to_numpy() == base)
        if not len(indices):
            continue
        cache_name = f"probability_{base}.npz"
        if cached(args, f"model_{base}", [cache_name, f"model_{base}_diagnostics.json"]):
            with np.load(args.work/cache_name) as saved:
                np.testing.assert_array_equal(saved["indices"], indices)
                probabilities[indices], classes[indices] = saved["probability"], saved["classes"]
            diagnostics[base] = json.loads((args.work/f"model_{base}_diagnostics.json").read_text())
            continue
        model, scaler, path = load_checkpoint(args, base, device)
        raw = gf.loc[indices, FEATURES[base]].to_numpy(np.float64)
        values = ((raw-np.asarray(scaler["mean"]))/np.asarray(scaler["scale"])).astype(np.float32)
        counts = collections.Counter()
        logger.info("DNA %s -> %s; %s sites; exactly 6 GF; batch=%s; device=%s", base, path.parent.name, len(indices), args.batch_size, device)
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                selected = indices[start:start+args.batch_size]
                sequences = [sequence_800(fasta, row) for row in sites.iloc[selected].itertuples()]
                encoded = tokenizer(sequences, truncation=False, padding="max_length", max_length=int(scaler["max_tokens"]), return_tensors="pt")
                if encoded.input_ids.shape[1] > int(scaler["max_tokens"]):
                    raise ValueError("800 bp exceeds trained token limit; refusing truncation")
                counts.update(map(int, encoded.attention_mask.sum(1)))
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else contextlib.nullcontext()
                with amp:
                    logits = model(encoded.input_ids.to(device), encoded.attention_mask.to(device),
                                   torch.from_numpy(values[start:start+len(selected)]).to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite model logits")
                probabilities[selected] = logits.float().softmax(-1)[:,1].cpu().numpy()
                classes[selected] = logits.argmax(-1).cpu().numpy()
                if start and start // args.batch_size % 500 == 0:
                    logger.info("Model %s: %s/%s", base, start+len(selected), len(indices))
        diag = dict(base=base, model_folder=path.parent.name, weight_sha256=sha256(path), feature_names=FEATURES[base],
                    feature_dim=6, strict_load=True, scaler_applied_once=True, sites=len(indices), token_counts=dict(counts),
                    truncated_rows=0, target_position_1based=401, precision=args.precision)
        temp = args.work / (cache_name+".tmp")
        with temp.open("wb") as f:
            np.savez(f, indices=indices, probability=probabilities[indices], classes=classes[indices])
        temp.replace(args.work/cache_name)
        save_json(args.work/f"model_{base}_diagnostics.json", diag)
        mark_cached(args, f"model_{base}", [cache_name, f"model_{base}_diagnostics.json"])
        diagnostics[base] = diag
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(args.work/"model_diagnostics.json", diagnostics)
    if not np.isfinite(probabilities).all() or not ((probabilities>=0)&(probabilities<=1)).all():
        raise ValueError("Invalid model probability")
    return probabilities, classes


def score_components(noes, pm, mouse, pig, probability):
    sequence = (np.asarray(mouse)+np.asarray(pig))/2
    probability = np.round(probability, 4)
    final = np.round((np.asarray(noes)+np.asarray(pm)+sequence+probability)/4, 4)
    return sequence, probability, final


def detailed_features(sites, gf):
    """Six named values per site, in its model's input order; retain GF status."""
    result = gf.drop(columns=["row_id"] + GF_NAMES).copy()
    for slot in range(6):
        name_column, value_column = f"gf_{slot+1}_name", f"gf_{slot+1}_value"
        result[name_column] = ""
        result[value_column] = np.nan
        for base, names in FEATURES.items():
            mask = sites.ref_base == base
            result.loc[mask, name_column] = names[slot]
            result.loc[mask, value_column] = gf.loc[mask, names[slot]].to_numpy(float)
    return result


def run(args):
    args.work.mkdir(parents=True, exist_ok=True)
    identity = resource_identity(args)
    path = args.work / "run_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Input/model/resource/code/options changed; use a fresh --work directory")
    else:
        if any(args.work.iterdir()):
            raise ValueError("Work directory is not empty and has no valid manifest; select a fresh directory")
        save_json(path, identity)
    human = Fasta(args.hg38_fa)
    try:
        aliases = aliases_for(human, args.assembly_report)
        original, sites = normalize_sites(args.user_df_path, human, aliases, args.invalid_sites, args.work)
        logger.info("Retained %s sites; exact strand-oriented reference routing", len(sites))
        if not cached(args, "reference", ["reference_scores.csv"]):
            save_table(reference_scores(args, sites, aliases), args.work/"reference_scores.csv")
            mark_cached(args, "reference", ["reference_scores.csv"])
        reference = read_table(args.work/"reference_scores.csv")
        if not cached(args, "alignment", ["alignment.csv"]):
            save_table(cross_species(args, sites, human), args.work/"alignment.csv")
            mark_cached(args, "alignment", ["alignment.csv"])
        alignment = read_table(args.work/"alignment.csv", keep_default_na=False)
        gf = generate_features(args, sites)
        probability, classes = predict(args, sites, gf, human)
        if len(reference) != len(sites) or len(alignment) != len(sites):
            raise ValueError("Score rows misaligned")
        sequence, rounded_probability, final = score_components(reference.noes, reference.pm,
                     alignment.seq_score_mm39, alignment.seq_score_sus11, probability)
        result = sites[["seqnames", "position", "strand"]].copy()
        result["modcon score"] = final
        outputs = {"summary": str(args.output_path)}
        if args.detailed_output:
            appended = pd.concat([reference, alignment, detailed_features(sites, gf)], axis=1)
            appended["seq_score"] = sequence
            appended["predicted_probability"] = rounded_probability
            appended["predicted_probability_raw"] = probability
            appended["predicted_class"] = classes
            appended["ref_base"] = sites.ref_base
            appended["model_base"] = sites.ref_base
            appended["model_folder"] = sites.ref_base.map(lambda x: model_path(args.base_model_dir,x).parent.name if x in "ACGT" else "")
            appended["prediction_status"] = np.where(sites.ref_base.isin(list("ACGT")), "ok", "unsupported_ref_base")
            appended["modcon_score"] = final
            appended["source_row_1based"] = sites.row_id
            appended["seq_800bp"] = [sequence_800(human,row) for row in sites.itertuples()]
            overlaps = set(original.columns) & set(appended.columns)
            if overlaps - {"ref_base"}:
                raise ValueError(f"Detailed output would overwrite input columns: {sorted(overlaps)}")
            if "ref_base" in overlaps:
                appended = appended.drop(columns="ref_base")
            detailed = pd.concat([original, appended], axis=1)
            for name in original:
                assert detailed[name].tolist() == original[name].tolist()
            args.detailed_output.parent.mkdir(parents=True, exist_ok=True)
            save_table(detailed, args.detailed_output)
            # Re-read every original field to catch separator/order/quoting mistakes.
            saved = read_table(args.detailed_output, dtype=str, keep_default_na=False)
            pd.testing.assert_frame_equal(saved[original.columns], original, check_dtype=False)
            outputs["detailed"] = str(args.detailed_output)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        save_table(result, args.output_path)
        saved = read_table(args.output_path)
        if saved.columns.tolist() != ["seqnames", "position", "strand", "modcon score"] or len(saved) != len(sites):
            raise ValueError("Output contract violated")
        np.testing.assert_allclose(saved["modcon score"], final, rtol=0, atol=1e-12)
        if sha256(args.user_df_path) != identity["resources"][str(args.user_df_path)]["sha256"]:
            raise ValueError("Input changed during execution")
        save_json(args.work/"completion.json", dict(status="complete", rows=len(sites), input=str(args.user_df_path), outputs=outputs,
                  output_sha256=sha256(args.output_path), features_per_model=FEATURES,
                  formula="round((noes+pm+(seq_score_mm39+seq_score_sus11)/2+round(probability,4))/4,4)",
                  source_strand_combined_with_chain=True, window_target_position_1based=401,
                  limitations=["R upstream standardization/clustering depends on the whole submitted mapped dataset.",
                               "Training-time upstream R parameters were not supplied; checkpoint scaler alone cannot remove this dataset dependence."],
                  resource_fingerprints="SHA256 for input/weights/code/chains/FAI; size+mtime_ns for FASTA and reference catalogues"))
        logger.info("COMPLETE: %s rows -> %s", len(sites), args.output_path)
    finally:
        human.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--user_df_path", "--input", type=Path, required=True)
    p.add_argument("--output_path", "--output", type=Path, default=Path("master_pipeline_output.csv"))
    p.add_argument("--detailed-output", type=Path, help="Preserve input fields/order and append scores, mappings, sequence and six named GF values for each site's model")
    p.add_argument("--work", type=Path, help="Checkpoint/log directory; default <output stem>_work")
    p.add_argument("--r_script_path", type=Path, default=HERE/"server_R.R")
    p.add_argument("--ontdata_path", "--ontdata", type=Path, required=True)
    p.add_argument("--mouse_raw_path", "--mouse", type=Path, default=Path("mouse_raw.csv"))
    p.add_argument("--pig_raw_path", "--pig", type=Path, default=Path("pig_raw.csv"))
    p.add_argument("--hg38_fa", "--hg38", type=Path, required=True)
    p.add_argument("--mm39_fa", "--mm39", type=Path, required=True)
    p.add_argument("--sus11_fa", "--pig-fasta", type=Path, required=True)
    p.add_argument("--mm39_chain", "--mm39-chain", type=Path, required=True)
    p.add_argument("--sus11_chain", "--pig-chain", type=Path, required=True)
    p.add_argument("--hg19_chain", "--hg19-chain", type=Path, default=Path("chain/hg38ToHg19.over.chain"))
    p.add_argument("--dnabert_model_path", "--dnabert", type=Path, default=Path("models/DNABERT-2-117M"))
    p.add_argument("--base_model_dir", "--models", type=Path, default=Path("models"))
    p.add_argument("--assembly-report", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--precision", choices=["fp32","bf16"], default="fp32")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--invalid-sites", choices=["error","drop"], default="error")
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if not args.work:
        args.work = args.output_path.with_name(args.output_path.stem+"_work")
    outputs = [args.output_path] + ([args.detailed_output] if args.detailed_output else [])
    inputs = [args.user_df_path, args.ontdata_path, args.mouse_raw_path, args.pig_raw_path, args.hg38_fa, args.mm39_fa,
              args.sus11_fa, args.hg19_chain, args.mm39_chain, args.sus11_chain, args.r_script_path]
    if len(set(outputs)) != len(outputs) or set(outputs)&set(inputs):
        p.error("Outputs must differ from each other and from all inputs/resources")
    for output in outputs:
        if output.suffix.lower() not in (".csv", ".tsv"):
            p.error("Outputs must use .csv or .tsv")
    for source in inputs:
        if not source.is_file():
            p.error(f"Missing input/resource: {source}")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
