"""
From https://academic.oup.com/bioinformatics/article/40/9/btae512/7737470:

To minimize potential artefacts in variant calling, we filter all variants found in repeat regions, according to RepeatMasker (Smit et al. 1996–2010). 
We filter all variants annotated as RNA edits. We remove all low coverage variants found with cov in <10% of the cells.

To allow flexibility in filtering of germline variants and low coverage variants, we create six different variant subsets corresponding to the combination of different thresholds. 
These thresholds are exclusion/inclusion of germline variants and exclusion of variants with MAF of 2%, 5%, or 10%. 
We then run the wNMF on each subset and later choose the most informative subset.
"""

rule create_filtered_bam:
    input:
        bam = rules.copy_bam_to_local.output.local_bam,
        cell_line_barcodes_file = os.path.join(OUTPUT_PATH, "results/mapping/cell_lines/{cell_line}/{cell_line}.barcodes.txt"),
        barcode_files_done = rules.create_cell_line_barcode_files.output.cell_line_barcode_files_done,
    output:
        out_dir = directory(os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/filtered/")),
        filtered_bam = os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/filtered/filtered.bam"),
        done = os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/filtered_done.txt"),
    log:
        stdout="logs/create_filtered_bam_{cell_line}.stdout",
        stderr="logs/create_filtered_bam_{cell_line}.stderr",
    shell:
        """
        mkdir -p {output.out_dir}
        samtools view -@8 -b -D BC:{input.cell_line_barcodes_file} {input.bam} | samtools sort -@8 -o {output.filtered_bam} >> {log.stdout} 2>> {log.stderr}
        samtools index {output.filtered_bam}
        touch {output.done}
        """

rule run_discover_cell_line_snps:
    input:
        bam = rules.create_filtered_bam.output.filtered_bam,
        done = rules.create_filtered_bam.output.done,
    output:
        out_dir = directory(os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/discovery/")),
        done = os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/discovery_done.txt"),
    params:
        reference_fasta = config["snv"]["reference_fasta"],
    log:
        stdout="logs/run_discover_cell_line_snps_{cell_line}.stdout",
        stderr="logs/run_discover_cell_line_snps_{cell_line}.stderr",
    threads: config["snv"]["threads"]["pileup"]
    conda:
        "../envs/snv.yaml",
    shell:
        """
        mkdir -p {output.out_dir}
        cellsnp-lite -s {input.bam} \
            -O {output.out_dir} -p {threads} \
            --minMAF 0.00 --minCOUNT 50 --refseq {params.reference_fasta} \
            --cellTAG None --UMItag "UB" --gzip >> {log.stdout} 2>> {log.stderr}
        touch {output.done}
        # For bulk data: the same but lower minCOUNT
        """

rule filter_variants:
    input:
        discovery_dir = rules.run_discover_cell_line_snps.output.out_dir,
        done = rules.run_discover_cell_line_snps.output.done,
    output:
        filtered_vcf = os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/cellSNP.filtered.vcf.gz"),
    params:
        reference_fasta = config["snv"]["reference_fasta"],
        min_alt_count = config["snv"]["min_alt_count"],
        min_depth = config["snv"]["min_depth"],
        min_vaf = config["snv"]["min_vaf"],
    log:
        stdout="logs/filter_variants_{cell_line}.stdout",
        stderr="logs/filter_variants_{cell_line}.stderr",
    shell:
        """
        bcftools norm -f {params.reference_fasta} -m -both -Oz -o normed.{wildcards.cell_line}.vcf.gz {input.discovery_dir}/cellSNP.renamed.vcf.gz
        bcftools index -f normed.{wildcards.cell_line}.vcf.gz
        # Also remove Ns...
        bcftools view -i 'INFO/AD>={params.min_alt_count} && INFO/DP>={params.min_depth} && INFO/AD/INFO/DP>={params.min_vaf}' normed.{wildcards.cell_line}.vcf.gz -Oz -o {output.filtered_vcf}
        tabix -p vcf {output.filtered_vcf}
        """

rule annotate_variants_per_cell_line:
    input:
        filtered_vcf = rules.filter_variants.output.filtered_vcf,
    output:
        tsv = os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/vep/{cell_line}_annotations.tsv"),
    params:
        rnaEditFile = config["snv"]["rnaEditFile"],
        dbfile = config["snv"]["dbfile"],
    log:
        stdout="logs/annotate_variants_per_cell_line_{cell_line}.stdout",
        stderr="logs/annotate_variants_per_cell_line_{cell_line}.stderr",
    threads: config["snv"]["threads"]["vep"]
    conda:
        "../envs/snv.yaml",
    shell:
        """
        vep -i {input.filtered_vcf} --cache --dir_cache \
            --symbol --tab -o {output.tsv} --force_overwrite --check_existing \
            --exclude_null_alleles --custom {params.rnaEditFile},REDI,vcf,exact --custom {params.dbfile},dbSNP-common,vcf,exact \
            --af --fields "Location,Allele,Gene,Feature,Feature_type,Consequence,Amino_acids,Codons,Existing_variation,IMPACT,REDI,AF,dbSNP-common" \
            --fork {threads} >> {log.stdout} 2>> {log.stderr}
        """

rule run_pileup_per_sample:
    input:
        bam = rules.create_filtered_bam.output.filtered_bam,
        filtered_vcf = rules.filter_variants.output.filtered_vcf,
        barcodes_file = lambda wildcards: os.path.join(OUTPUT_PATH, "results/mapping/cell_lines/{cell_line}/{split_key}/barcodes/{sample}.barcodes.txt"),
    output:
        pileup_dir = directory(os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/{split_key}/{sample}/pileup/")),
        pileup_done = os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/{split_key}/{sample}/pileup_done.txt"),
    log:
        stdout="logs/run_pileup_per_sample_{cell_line}_{split_key}_{sample}.stdout",
        stderr="logs/run_pileup_per_sample_{cell_line}_{split_key}_{sample}.stderr",
    threads: config["snv"]["threads"]["pileup"]
    conda:
        "../envs/snv.yaml",
    shell:
        """
        gzip -dkf {input.filtered_vcf}
        cellsnp-lite -s {input.bam} -b {input.barcodes_file} -O {output.pileup_dir} -p {threads} \
            -R {input.filtered_vcf} \
            --cellTAG "BC" --UMItag "UB" --gzip >> {log.stdout} 2>> {log.stderr}
        touch {output.pileup_done}
        """

# vep -i /mnt/data/pedro/P28304/results/snv/A549/cellSNP.filtered.vcf.gz --cache --dir_cache --symbol --tab -o /mnt/data/pedro/P28304/results/snv/A549/vep/A549_annotations.tsv --force_overwrite --check_existing             --exclude_null_alleles --custom /home/pedro/projects/cancer/clonal_drug_response/pipeline/ss3/resources/TABLE1_hg38_v2.txt.gz,REDI,vcf,exact --custom /home/pedro/projects/cancer/clonal_drug_response/pipeline/ss3/resources/00-common_all_chr.vcf.gz,dbSNP-common,vcf,exact             --af --fields "Location,Allele,Gene,Feature,Feature_type,Consequence,Amino_acids,Codons,Existing_variation,IMPACT,REDI,AF,dbSNP-common" --fork 8 >> logs/annotate_variants_per_cell_line_A549.stdout 2>> logs/annotate_variants_per_cell_line_A549.stderr


rule create_snv_anndata_per_sample:
    input:
        pileup_dir = rules.run_pileup_per_sample.output.pileup_dir,
        variant_annotations = rules.annotate_variants_per_cell_line.output.tsv,
    output:
        anndata = os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/{split_key}/{sample}/anndata/{sample}.anndata.h5ad"),
    params:
        scripts_path = config["scripts_path"],
        perc = config["snv"]["anndata"]["perc"],
        filter_hela = config["snv"]["anndata"]["filter_hela"],
    log:
        stdout="logs/create_snv_anndata_per_sample_{cell_line}_{split_key}_{sample}.stdout",
        stderr="logs/create_snv_anndata_per_sample_{cell_line}_{split_key}_{sample}.stderr",
    conda:
        "../envs/snv.yaml",
    shell:
        """
        python {params.scripts_path}/create_snv_anndata.py -d {input.pileup_dir} -a {input.variant_annotations} -o {output.anndata} \
            -p {params.perc} -f {params.filter_hela} >> {log.stdout} 2>> {log.stderr}
        """

def aggregate_snv_samples(wildcards):
    checkpoint_output = checkpoints.create_sample_barcode_files.get(**wildcards).output[0]
    return expand(
        os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/{split_key}/{sample}/anndata/{sample}.anndata.h5ad"),
        cell_line=wildcards.cell_line,
        split_key=wildcards.split_key,
        sample=glob_wildcards(os.path.join(checkpoint_output, "{sample}.barcodes.txt")).sample
    )

rule merge_snv_anndata:
    input:
        anndata = aggregate_snv_samples,
    output:
        anndata = os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/{split_key}/anndata/{cell_line}.anndata.h5ad"),
    params:
        scripts_path = config["scripts_path"],
    log:
        stdout="logs/merge_snv_anndata_{cell_line}_{split_key}.stdout",
        stderr="logs/merge_snv_anndata_{cell_line}_{split_key}.stderr",
    conda:
        "../envs/snv.yaml",
    shell:
        """
        python {params.scripts_path}/merge_snv_anndata.py -i {input.anndata} -o {output.anndata} >> {log.stdout} 2>> {log.stderr}
        """ 

rule run_snv:
    input:
        anndata = rules.merge_snv_anndata.output.anndata,
    output:
        done=os.path.join(OUTPUT_PATH, "results/snv/{cell_line}/{split_key}/snv_done.txt"),
    log:
        stdout="logs/run_snv_{cell_line}_{split_key}.stdout",
        stderr="logs/run_snv_{cell_line}_{split_key}.stderr",
    shell:
        """
        echo "All SNV samples completed for {wildcards.cell_line}_{wildcards.split_key}" >> {log.stdout} 2>> {log.stderr}
        touch {output.done}
        """
