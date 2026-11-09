# Usage: Rscript server_R.R <sites.csv> <features.csv> <hg38ToHg19.chain>
# Python provides validated hg38 1-based single-base sites, stable row_id and ref_base.
# One annotation call across the complete mapped input preserves the original
# predictors_annot standardization/clustering semantics (never by GPU batch/base).
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop("Usage: Rscript server_R.R sites.csv features.csv hg38ToHg19.chain")
input <- args[[1]]; output <- args[[2]]; chain_path <- args[[3]]
suppressPackageStartupMessages({
  library(GenomicRanges)
  library(SummarizedExperiment)
  library(rtracklayer)
  library(TxDb.Hsapiens.UCSC.hg19.knownGene)
  library(BSgenome.Hsapiens.UCSC.hg19)
  library(m6ALogisticModel)
})
GF <- c("UTR5","UTR3","cds","Stop_codons","Start_codons","TSS","TSS_A",
 "exon_stop","alternative_exon","constitutive_exon","internal_exon","long_exon",
 "last_exon","last_exon_400bp","last_exon_sc400","intron","pos_UTR5","pos_UTR3",
 "pos_cds","pos_exons","dist_sj_5_p2000","dist_sj_3_p2000","length_UTR3","length_UTR5",
 "length_cds","length_gene_ex","length_gene_full","length_tx_exon","length_tx_full",
 "clust_f1000","clust_f100","dist_nearest_p2000","dist_nearest_p200","sncRNA","lncRNA",
 "isoform_num","exon_num","GC_cont_genes","GC_cont_101bp","GC_cont_101bp_abs")
FEATURES <- list(
 A=c("length_tx_full","GC_cont_genes","clust_f100","pos_exons","length_cds","exon_num"),
 C=c("clust_f100","long_exon","GC_cont_101bp","dist_sj_3_p2000","length_gene_ex","length_cds"),
 G=c("length_tx_full","exon_stop","length_cds","cds","exon_num","clust_f100"),
 T=c("long_exon","dist_sj_3_p2000","length_tx_full","GC_cont_genes","clust_f100","UTR3"))
sites <- read.csv(input, stringsAsFactors=FALSE, check.names=FALSE)
stopifnot(all(c("row_id","seqnames","position","strand","ref_base") %in% names(sites)),
          !anyDuplicated(sites$row_id), nrow(sites)>0L,
          all(sites$position>=1 & sites$position==floor(sites$position)),
          all(sites$strand %in% c("+","-")))
dir.create(dirname(output), recursive=TRUE, showWarnings=FALSE)
gr <- GRanges(sites$seqnames, IRanges(sites$position, width=1), strand=sites$strand,
              row_id=sites$row_id)
chain <- import.chain(chain_path, exclude="__NO_CONTIG_MATCHES_THIS_FILTER__")
lifted <- liftOver(gr, chain)
counts <- elementNROWS(lifted)
first <- unlist(lifted[counts>0L], use.names=FALSE)
first <- first[!duplicated(first$row_id)]
where <- match(first$row_id, sites$row_id)
limits <- seqlengths(BSgenome.Hsapiens.UCSC.hg19::Hsapiens)
has_ref <- as.character(seqnames(first)) %in% names(limits)
safe <- has_ref & start(first)>=51L & start(first)<=limits[as.character(seqnames(first))]-50L
safe[is.na(safe)] <- FALSE
meta <- data.frame(row_id=sites$row_id, hg19_mapping_count=counts,
                   hg19_seqnames="", hg19_position=NA_integer_, hg19_strand="", gf_status="unmapped")
meta$hg19_seqnames[where] <- as.character(seqnames(first))
meta$hg19_position[where] <- start(first)
meta$hg19_strand[where] <- as.character(strand(first))
meta$gf_status[where] <- ifelse(!has_ref,"missing_hg19_reference",
                               ifelse(safe,"ok","hg19_window_out_of_bounds"))
gr <- first[safe]
genome(gr) <- "hg19"
features <- as.data.frame(matrix(0, nrow(sites), length(GF)))
names(features) <- GF
missing <- setNames(integer(length(GF)), GF)
if (length(gr)>0L) {
  se <- SummarizedExperiment(rowRanges=gr)
  annotated <- predictors_annot(se=se, txdb=TxDb.Hsapiens.UCSC.hg19.knownGene,
    bsgnm=BSgenome.Hsapiens.UCSC.hg19::Hsapiens,
    isoform_ambiguity_method="longest_tx", genes_ambiguity_method="average",
    annot_clustering=se, standardization=TRUE)
  f <- as.data.frame(mcols(annotated))[,GF,drop=FALSE]
  f[] <- lapply(f, as.numeric)
  missing <- vapply(f, function(x) sum(is.na(x)), integer(1))
  # Small/constant datasets can produce NaN z-scores: match legacy NA->0.
  f[is.na(f)] <- 0
  if (!all(is.finite(as.matrix(f)))) stop("Nonfinite genomic feature after NA fill")
  features[match(gr$row_id, sites$row_id),] <- f
}
result <- cbind(meta,features)
write.csv(data.frame(feature=GF,na_before_fill=missing),
          file.path(dirname(output),"feature_missing_counts.csv"),row.names=FALSE)
tmp <- paste0(output,".tmp")
write.csv(result,tmp,row.names=FALSE,na="")
if (!file.rename(tmp,output)) stop("Cannot publish R feature output")
# Exactly six ordered numeric columns per model; row_id is an identifier, not a feature.
# DNA T is published as model_U. These are upstream R-transformed values; Python
# applies the checkpoint's additional train-fitted mean/scale exactly once.
for (base in names(FEATURES)) {
  ix <- which(sites$ref_base==base)
  write.csv(cbind(row_id=sites$row_id[ix],features[ix,FEATURES[[base]],drop=FALSE]),
            file.path(dirname(output),paste0("model_features_",ifelse(base=="T","U",base),".csv")),row.names=FALSE)
}
writeLines(capture.output(sessionInfo()),file.path(dirname(output),"r_session_info.txt"))
cat("FEATURES COMPLETE:",nrow(sites),"rows;",length(gr),"annotated; 6 selected per model\n")
