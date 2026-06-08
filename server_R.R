# ==============================================================================
# Module 4: Genomic Feature Extraction & Sequence Generation Line (Silent Edition)
# ==============================================================================

# Suppress annoying startup packages loading warnings
suppressMessages(suppressWarnings({
  library(TxDb.Hsapiens.UCSC.hg19.knownGene)
  library(SummarizedExperiment)
  library(BSgenome.Hsapiens.UCSC.hg19)
  library(BSgenome.Hsapiens.UCSC.hg38)       
  library(fitCons.UCSC.hg19)
  library(phastCons100way.UCSC.hg19)
  library(GenomicRanges)                     
  library(rtracklayer)                       
  library(e1071)
  library(ROCR)
  library(pROC)
  library(dplyr)
  library(caret)
  library(m6ALogisticModel) 
}))

# Parse arguments passed from Python
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage Error! Correct usage: Rscript server_R.R <input_path> <output_path>", call. = FALSE)
}

input_path <- args[1]
output_path <- args[2]

user_sites <- read.csv(input_path)
user_sites$id <- 1:nrow(user_sites)

# Extract 800bp flanking hg38 sequence coordinates
hg38_seq_gr <- GRanges(
  seqnames = user_sites$seqnames,
  ranges = IRanges(start = user_sites$position - 400, end = user_sites$position + 399),
  strand = user_sites$strand
)

# seqinfo(hg38_seq_gr) <- seqinfo(BSgenome.Hsapiens.UCSC.hg38::Hsapiens)
# hg38_seq_gr <- trim(hg38_seq_gr)
user_sites$seq_800bp <- as.character(getSeq(BSgenome.Hsapiens.UCSC.hg38::Hsapiens, hg38_seq_gr))

# Convert Human coordinates from hg38 to hg19 via liftOver
hg38_gr <- GRanges(
  seqnames = user_sites$seqnames,
  ranges = IRanges(start = user_sites$position, end = user_sites$position),
  strand = user_sites$strand,
  id = user_sites$id
)

chain <- import.chain('~/consRM2/chain_file/hg38ToHg19.over.chain')
user_sites_hg19 <- unlist(liftOver(hg38_gr, chain))
genome(user_sites_hg19) <- "hg19"

# Feature Annotation Engine Module 
GF_generation <- function(data){
  analysis_data <- data
  matureSE <- SummarizedExperiment()
  rowRanges(matureSE) <- analysis_data 
  
  # Suppress messages generated inside internal library calls
  suppressMessages(suppressWarnings({
    data_standardized <- predictors_annot(
      se = matureSE,
      txdb = TxDb.Hsapiens.UCSC.hg19.knownGene,
      bsgnm = BSgenome.Hsapiens.UCSC.hg19::Hsapiens,
      isoform_ambiguity_method = "longest_tx",
      genes_ambiguity_method = "average",
      annot_clustering = matureSE,
      standardization = TRUE
    )
  }))
  
  GF <- mcols(data_standardized)
  return(GF)
}

gr_features_raw <- GF_generation(user_sites_hg19)

GF_names <- c(
  "UTR5","UTR3","cds" , "Stop_codons", "Start_codons",
  "TSS","TSS_A","exon_stop","alternative_exon","constitutive_exon",
  "internal_exon","long_exon","last_exon","last_exon_400bp",
  "last_exon_sc400","intron","pos_UTR5","pos_UTR3","pos_cds",
  "pos_exons","dist_sj_5_p2000","dist_sj_3_p2000","length_UTR3",
  "length_UTR5","length_cds","length_gene_ex","length_gene_full",
  "length_tx_exon",  "length_tx_full","clust_f1000" , "clust_f100",
  "dist_nearest_p2000","dist_nearest_p200","sncRNA","lncRNA","isoform_num",
  "exon_num","GC_cont_genes", "GC_cont_101bp" ,"GC_cont_101bp_abs"
)
gr_features <- as.data.frame(gr_features_raw)[, GF_names, drop = FALSE]
gr_features$id <- user_sites_hg19$id

# Data curation cleanups
gr_features[] <- lapply(gr_features, function(x) {
  if (is.logical(x)) return(as.numeric(x))
  return(x)
})
gr_features[is.na(gr_features)] <- 0

# Merge features back into master hg38 data structure
final_df <- merge(user_sites, gr_features, by = "id", all.x = TRUE)

for(col in GF_names) {
  final_df[[col]][is.na(final_df[[col]])] <- 0
}

final_df$id <- NULL

# Write data out quietly
write.csv(final_df, output_path, row.names = FALSE)
