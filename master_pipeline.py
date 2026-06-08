#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ModCon Integrated Multimodal Conservation Scoring Pipeline (CLI Production Version)
Modules integrated: Base Library Mapping -> Cross-Species Alignment -> R Genomic Features -> DL Inference
================================================================================
"""

import os
import sys
import subprocess
import tempfile
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from Bio import SeqIO
from Bio.Seq import Seq
from liftover import ChainFile

# Setup production logging environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Global Constants & Scoring Matrix ====================
BASE_INDEX = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
SM = np.array([
    [2, 1, 0, 0],  # A
    [1, 2, 0, 0],  # G
    [0, 0, 2, 1],  # C
    [0, 0, 1, 2]   # T
])

GF_NAMES = [
    "UTR5","UTR3","cds" , "Stop_codons", "Start_codons",
    "TSS","TSS_A","exon_stop","alternative_exon","constitutive_exon",
    "internal_exon","long_exon","last_exon","last_exon_400bp",
    "last_exon_sc400","intron","pos_UTR5","pos_UTR3","pos_cds",
    "pos_exons","dist_sj_5_p2000","dist_sj_3_p2000","length_UTR3",
    "length_UTR5","length_cds","length_gene_ex","length_gene_full",
    "length_tx_exon",  "length_tx_full","clust_f1000" , "clust_f100",
    "dist_nearest_p2000","dist_nearest_p200","sncRNA","lncRNA","isoform_num",
    "exon_num","GC_cont_genes", "GC_cont_101bp" ,"GC_cont_101bp_abs"
]

# ==================== Base Feature Processing Components ====================
def calculate_alignment_score(seq1, seq2):
    """Calculate 11bp sequence alignment score and normalize it (/32)"""
    if len(seq1) != 11 or len(seq2) != 11:
        return 0.0
    score = 0
    for b1, b2 in zip(seq1.upper(), seq2.upper()):
        if b1 in BASE_INDEX and b2 in BASE_INDEX:
            score += SM[BASE_INDEX[b1], BASE_INDEX[b2]]
    return score / 32.0

def merge_reference_feature(df, ontdata, ref_col_name, target_col_name, default_value=0.0):
    """Generic mapping function to merge reference database features"""
    ontdata_subset = ontdata[['seqnames', 'position', 'strand', ref_col_name]].copy()
    ontdata_subset[ref_col_name] = pd.to_numeric(ontdata_subset[ref_col_name], errors='coerce')
    merged_df = pd.merge(df, ontdata_subset, on=['seqnames', 'position', 'strand'], how='left')
    merged_df = merged_df.rename(columns={ref_col_name: target_col_name})
    merged_df[target_col_name] = merged_df[target_col_name].fillna(default_value).astype(float)
    return merged_df

def compute_single_species_alignment(row, hg38_dict, target_dict, converter, conserved_set):
    """Compute cross-species sequence similarity and conservation status based on strand direction"""
    chrom, pos_1based, strand = str(row['seqnames']), int(row['position']), str(row['strand'])
    start_0based, end_0based = pos_1based - 6, pos_1based + 5
    
    try:
        if chrom not in hg38_dict or converter is None:
            return 0.0, '-'
            
        human_11bp_str = str(hg38_dict[chrom].seq[start_0based:end_0based]).upper()
        human_center_base = str(hg38_dict[chrom].seq[pos_1based - 1]).upper()
        
        if strand == '-':
            human_11bp_str = str(Seq(human_11bp_str).reverse_complement()).upper()
            human_center_base = str(Seq(human_center_base).reverse_complement()).upper()
        
        lifted_targets = converter.query(chrom, pos_1based - 1)
        if not lifted_targets:
            return 0.0, '-'
            
        target_chrom, target_pos_0based, target_strand = lifted_targets[0]
        if target_chrom not in target_dict:
            return 0.0, '-'
            
        target_center_base = str(target_dict[target_chrom].seq[target_pos_0based]).upper()
        if target_strand == '-':
            target_center_base = str(Seq(target_center_base).reverse_complement()).upper()

        if target_center_base != human_center_base:
            return 0.0, '-'

        target_start_0based, target_end_0based = target_pos_0based - 5, target_pos_0based + 6
        target_11bp_str = str(target_dict[target_chrom].seq[target_start_0based:target_end_0based]).upper()
        if target_strand == '-':
            target_11bp_str = str(Seq(target_11bp_str).reverse_complement()).upper()

        row_score = calculate_alignment_score(human_11bp_str, target_11bp_str)
        target_pos_1based = target_pos_0based + 1
        site_pos_target = f"{target_chrom}:{target_pos_1based} {strand}"
        status = 'conserved' if site_pos_target in conserved_set else '-'

        return round(row_score, 2), status
    except Exception:
        return 0.0, '-'

# ==================== Deep Learning Framework Components ====================
class DNAInferenceDataset(Dataset):
    def __init__(self, sequences, genomic_features, tokenizer, max_length=200):
        self.sequences = sequences
        self.genomic_features = genomic_features
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        genomic_feat = torch.tensor(self.genomic_features[idx], dtype=torch.float32)
        encoding = self.tokenizer(sequence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'genomic_features': genomic_feat}

class DNABERT2ConservationModel(nn.Module):
    def __init__(self, model_path, genomic_feature_dim=40, num_classes=2, hidden_dim=256, dropout_rate=0.1):
        super(DNABERT2ConservationModel, self).__init__()
        self.dnabert2 = BertModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.dnabert2_dim = self.dnabert2.config.hidden_size
        self.genomic_proj = nn.Sequential(
            nn.Linear(genomic_feature_dim, max(64, genomic_feature_dim * 16)), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(max(64, genomic_feature_dim * 16), 128), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        fusion_dim = self.dnabert2_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, input_ids, attention_mask, genomic_features):
        dnabert_outputs = self.dnabert2(input_ids=input_ids, attention_mask=attention_mask)
        sequence_features = dnabert_outputs.last_hidden_state[:, 0, :]
        genomic_processed = self.genomic_proj(genomic_features)
        return self.classifier(torch.cat([sequence_features, genomic_processed], dim=1))

# ==================== Core Master Pipeline Engine ====================
def run_master_pipeline(args):
    # ----------------- Step 1: Base Library Feature Mapping -----------------
    logger.info("[STEP 1/5] Extracting base library features (Within-species Recurrence & Orthologous Modification Concordance)...")
    df = pd.read_csv(args.user_df_path, dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=['seqnames', 'position', 'strand'])
    df['position'] = df['position'].astype(float).astype(int)
    
    ontdata = pd.read_csv(args.ontdata_path, dtype=str)
    ontdata.columns = ontdata.columns.str.strip().str.lower()
    ontdata['position'] = ontdata['start'].astype(float).astype(int)
    
    df = merge_reference_feature(df, ontdata, 'rank_noes', 'noes')
    df = merge_reference_feature(df, ontdata, 'pm_score', 'pm')
    
    # ----------------- Step 2: Cross-Species Alignment -----------------
    logger.info("[STEP 2/5] Loading reference genomes and executing Sequence-context Conservation...")
    hg38_dict = SeqIO.index(args.hg38_fa, "fasta")
    mm39_dict = SeqIO.index(args.mm39_fa, "fasta")
    sus11_dict = SeqIO.index(args.sus11_fa, "fasta")
    mm39_lifter = ChainFile(args.mm39_chain, 'hg38', 'mm39')
    sus11_lifter = ChainFile(args.sus11_chain, 'hg38', 'susScr11')
    
    mm39_conserved_set = set(pd.read_csv(args.mouse_raw_path, usecols=['site_pos_mm39'])['site_pos_mm39'].astype(str).str.strip()) if os.path.exists(args.mouse_raw_path) else set()
    sus11_conserved_set = set(pd.read_csv(args.pig_raw_path, usecols=['site_pos_sus11'])['site_pos_sus11'].astype(str).str.strip()) if os.path.exists(args.pig_raw_path) else set()
    
    mm39_scores, mm39_statuses, sus11_scores, sus11_statuses = [], [], [], []
    for idx, row in df.iterrows():
        s_m, st_m = compute_single_species_alignment(row, hg38_dict, mm39_dict, mm39_lifter, mm39_conserved_set)
        s_s, st_s = compute_single_species_alignment(row, hg38_dict, sus11_dict, sus11_lifter, sus11_conserved_set)
        mm39_scores.append(s_m); mm39_statuses.append(st_m)
        sus11_scores.append(s_s); sus11_statuses.append(st_s)
        
    df['seq_score_mm39'], df['status_mm39'] = mm39_scores, mm39_statuses
    df['seq_score_sus11'], df['status_sus11'] = sus11_scores, sus11_statuses
    
    # ----------------- Step 3: R Subprocess Command -----------------
    logger.info("[STEP 3/5] Executing R script to generate genomic features & reference sequences...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = os.path.join(tmpdir, "p_to_r.csv")
        tmp_out = os.path.join(tmpdir, "r_to_p.csv")
        df.to_csv(tmp_in, index=False)
        
        result = subprocess.run(["Rscript", args.r_script_path, tmp_in, tmp_out], capture_output=True, text=True)
        if result.returncode != 0: 
            logger.error(f"R script execution failed with exit code {result.returncode}!")
            if result.stderr: print(result.stderr)
            raise RuntimeError(f"R script crashed: {result.stderr}")
            
        processed_df = pd.read_csv(tmp_out)
        
    # ----------------- Step 4: PyTorch Multimodal Inference -----------------
    logger.info("[STEP 4/5] Running base-specific routing inference via Finetuned deep learning model")
    if 'seq_800bp' in processed_df.columns:
        processed_df = processed_df.rename(columns={'seq_800bp': 'sequence'})
        
    processed_df['center_base'] = processed_df['sequence'].str[400].str.upper()
    processed_df['predicted_class'] = 0
    processed_df['predicted_probability'] = 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.dnabert_model_path, trust_remote_code=True, local_files_only=True)
    
    for base in ['A', 'C', 'G', 'T']:
        base_mask = processed_df['center_base'] == base
        sub_df = processed_df[base_mask]
        if len(sub_df) == 0: continue
            
        logger.info(f" -> Forwarding {len(sub_df)} sites to base model [model_{base}]...")
        model_weights_path = os.path.join(args.base_model_dir, f"model_{base}", "best_mlp_model.pth")
        
        if not os.path.exists(model_weights_path):
            logger.warning(f"CRITICAL WARNING: Cannot find model weights at {model_weights_path}, skipping prediction for base {base}!")
            continue
            
        raw_features = sub_df[GF_NAMES].values.astype(np.float32)
        sequences = sub_df['sequence'].values
        
        dataloader = DataLoader(DNAInferenceDataset(sequences, raw_features, tokenizer), batch_size=4, shuffle=False, num_workers=2)
        model = DNABERT2ConservationModel(model_path=args.dnabert_model_path, genomic_feature_dim=40)
        
        checkpoint = torch.load(model_weights_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        
        probs_list, preds_list = [], []
        with torch.no_grad():
            for batch in dataloader:
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['genomic_features'].to(device))
                probs_list.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                preds_list.extend(torch.argmax(logits, dim=1).cpu().numpy())
                
        processed_df.loc[base_mask, 'predicted_probability'] = np.round(probs_list, 4)
        processed_df.loc[base_mask, 'predicted_class'] = preds_list

    # ----------------- Step 5: Format Reshaping & Saving -----------------
    logger.info("[STEP 5/5] Calculating final ensemble modcon score and filtering output columns...")
    processed_df = processed_df.drop(columns=['center_base'])
    
    # Force numerical data validation on component score columns to avoid unexpected string mean failures
    score_columns = ['noes', 'pm', 'seq_score_mm39', 'seq_score_sus11', 'predicted_probability']
    for col in score_columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0.0)
        
    # Compute the comprehensive arithmetic mean across all 5 scoring dimensions
    processed_df['modcon score'] = processed_df[score_columns].mean(axis=1).round(4)
    
    # Reshape table to exclusively preserve original keys alongside the composite score
    output_columns = ['seqnames', 'position', 'strand', 'modcon score']
    final_output = processed_df[output_columns]
    
    final_output.to_csv(args.output_path, index=False)
    logger.info(f"🎉 [SUCCESS] Pipeline execution complete! Filtered layout saved to: {args.output_path}")

# ==================== CLI Parser Architecture ====================
def main():
    parser = argparse.ArgumentParser(description="ModCon Integrated Multimodal Conservation Pipeline (CLI Production Version)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Core I/O Data paths
    parser.add_argument('--user_df_path', type=str, required=True, help='Path to user-uploaded modification sites file (CSV format)')
    parser.add_argument('--r_script_path', type=str, default='server_R.R', help='Path to companion R feature extraction engine script')
    parser.add_argument('--ontdata_path', type=str, required=True, help='Path to base reference library (ontdata.csv)')
    parser.add_argument('--output_path', type=str, default='master_pipeline_output.csv', help='Output file destination path')
    
    # Cross-Species Annotation Libraries
    parser.add_argument('--mouse_raw_path', type=str, default='mouse_raw.csv', help='Path to mouse known conserved coordinates catalog')
    parser.add_argument('--pig_raw_path', type=str, default='pig_raw.csv', help='Path to pig known conserved coordinates catalog')
    
    # Genome FASTA Assets & Conversion Chains
    parser.add_argument('--hg38_fa', type=str, required=True, help='Path to Human hg38 reference genome FASTA')
    parser.add_argument('--mm39_fa', type=str, required=True, help='Path to Mouse mm39 reference genome FASTA')
    parser.add_argument('--sus11_fa', type=str, required=True, help='Path to Pig susScr11 reference genome FASTA')
    parser.add_argument('--mm39_chain', type=str, required=True, help='Path to hg38ToMm39 conversion over.chain file')
    parser.add_argument('--sus11_chain', type=str, required=True, help='Path to hg38ToSusScr11 conversion over.chain file')
    
    # Deep Learning Directory Configurations
    parser.add_argument('--dnabert_model_path', type=str, default='./embedded_model', help='Path to local pre-trained DNABERT-2 workspace directory')
    parser.add_argument('--base_model_dir', type=str, default='./', help='Parent workspace directory containing model_A/C/G/U folders')

    args = parser.parse_args()
    
    logging.info("Scanning infrastructure assets and essential files...")
    checklist = [args.user_df_path, args.r_script_path, args.ontdata_path, args.hg38_fa, args.mm39_fa, args.sus11_fa, args.mm39_chain, args.sus11_chain, args.dnabert_model_path]
    
    for path in checklist:
        if not os.path.exists(path):
            logging.error(f"CRITICAL ERROR: Resource missing at target destination location -> {path}")
            sys.exit(1)
            
    print("✅ Infrastructure pre-flight check successful. Initializing multimodal services...\n")
    run_master_pipeline(args)

if __name__ == "__main__":
    main()
