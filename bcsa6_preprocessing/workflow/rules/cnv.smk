import glob
import os

rule run_numbat_preprocessing_per_sample:
    input:
        bam=rules.copy_bam_to_local.output.local_bam,
        barcodes_file = lambda wildcards: os.path.join(OUTPUT_PATH, "results/mapping/cell_lines/{cell_line}/{split_key}/barcodes/{sample}.barcodes.txt"),
        barcode_files_done = rules.create_sample_barcode_files.output.barcode_files_done,
    output:
        output_dir=directory(os.path.join(OUTPUT_PATH, "results/cnv/{cell_line}/{split_key}/{sample}/numbat_preprocessing/")),
        allele_counts=os.path.join(OUTPUT_PATH, "results/cnv/{cell_line}/{split_key}/{sample}/numbat_preprocessing/{sample}_allele_counts.tsv.gz"),
    params:
        gmap=config["cnv"]["gmap"],
        snpvcf=config["cnv"]["snp_vcf"],
        paneldir=config["cnv"]["reference_panel"],
        numbat_path=config["cnv"]["numbat_path"],
        eagle_path=config["cnv"]["eagle_path"],
        threads=8,
    log:
        stdout="logs/numbat_preprocessing_{cell_line}_{split_key}_{sample}.stdout",
        stderr="logs/numbat_preprocessing_{cell_line}_{split_key}_{sample}.stderr",
    conda:
        "cnv"  # use named environment because Numbat is not available through conda
    shell:
        """        
        # Run Numbat preprocessing for each sample
        mkdir -p {output.output_dir}

        Rscript {params.numbat_path}/numbat/inst/bin/pileup_and_phase.R \
            --label {wildcards.sample} \
            --samples {wildcards.sample} \
            --bams {input.bam} \
            --barcodes {input.barcodes_file} \
            --outdir {output.output_dir} \
            --gmap {params.gmap} \
            --snpvcf {params.snpvcf} \
            --paneldir {params.paneldir} \
            --ncores {params.threads} \
            --eagle {params.eagle_path} \
            --cellTAG "BC" >> {log.stdout} 2>> {log.stderr}

        # Create a done file to indicate preprocessing is complete
        touch {output.output_dir}/done.txt
        """

rule run_numbat_per_sample:
    """
    Run Numbat for a single sample
    """
    input:
        allele_counts=rules.run_numbat_preprocessing_per_sample.output.allele_counts,
        umi_counts=rules.make_filtered_adata.output.counts_csv,
    output:
        sample_done=os.path.join(OUTPUT_PATH, "results/cnv/{cell_line}/{split_key}/numbat_out/{sample}/numbat_done.txt"),
    params:
        output_path=OUTPUT_PATH,
        threshold=config["cnv"]["threshold"],
        gamma=config["cnv"]["gamma"],
        min_cells=config["cnv"]["min_cells"],
        min_LLR=config["cnv"]["min_LLR"],
        max_entropy=config["cnv"]["max_entropy"],
        n_cut=config["cnv"]["n_cut"],
        tau=config["cnv"]["tau"],
        init_k=config["cnv"]["init_k"],
        max_iter=config["cnv"]["max_iter"],
        check_convergence=config["cnv"]["check_convergence"],
        call_clonal_loh=config["cnv"]["call_clonal_loh"],
        segs_consensus_fix=config["cnv"]["segs_consensus_fix"],
        scripts_path=config["scripts_path"],
        genome=config["cnv"]["genome"],
        threads=8,
    log:
        stdout="logs/numbat_{cell_line}_{split_key}_{sample}.stdout",
        stderr="logs/numbat_{cell_line}_{split_key}_{sample}.stderr",
    conda:
        "cnv"  # use named environment because Numbat is not available through conda
    shell:
        """
        mkdir -p {params.output_path}/results/cnv/{wildcards.cell_line}/{wildcards.split_key}/numbat_out/{wildcards.sample}
        Rscript {params.scripts_path}/run_numbat.R \
            --allele_data {input.allele_counts} \
            --umi_counts {input.umi_counts} \
            --genome {params.genome} \
            --output {params.output_path}/results/cnv/{wildcards.cell_line}/{wildcards.split_key}/numbat_out/{wildcards.sample} \
            --threshold {params.threshold} \
            --gamma {params.gamma} \
            --min_cells {params.min_cells} \
            --min_LLR {params.min_LLR} \
            --max_entropy {params.max_entropy} \
            --n_cut {params.n_cut} \
            --tau {params.tau} \
            --init_k {params.init_k} \
            --max_iter {params.max_iter} \
            --check_convergence {params.check_convergence} \
            --call_clonal_loh {params.call_clonal_loh} \
            --segs_consensus_fix {params.segs_consensus_fix} \
            --ncores {params.threads} >> {log.stdout} 2>> {log.stderr}
        touch {output.sample_done}
        """

rule create_cnv_anndata_per_sample:
    input:
        numbat_out = rules.run_numbat_per_sample.output.sample_done,
    output:
        anndata = os.path.join(OUTPUT_PATH, "results/cnv/{cell_line}/{split_key}/{sample}/anndata/{sample}.anndata.h5ad"),
    params:
        scripts_path = config["scripts_path"],
    log:
        stdout="logs/create_cnv_anndata_{cell_line}_{split_key}_{sample}.stdout",
        stderr="logs/create_cnv_anndata_{cell_line}_{split_key}_{sample}.stderr",
    conda:
        "../envs/cnv.yaml",
    shell:
        """
        python {params.scripts_path}/create_cnv_anndata.py -i {params.output_path}/results/cnv/{wildcards.cell_line}/{wildcards.split_key}/numbat_out/{wildcards.sample}/ -o {output.anndata} >> {log.stdout} 2>> {log.stderr}
        """

def aggregate_cnv_samples(wildcards):
    checkpoint_output = checkpoints.create_sample_barcode_files.get(**wildcards).output[0]
    return expand(os.path.join(OUTPUT_PATH, "results/cnv/{cell_line}/{split_key}/numbat_out/{sample}/numbat_done.txt"),
           cell_line=wildcards.cell_line,
           split_key=wildcards.split_key,
           sample=glob_wildcards(os.path.join(checkpoint_output, "{sample}.barcodes.txt")).sample)


rule run_numbat:
    """
    Aggregate Numbat results from all samples
    """
    input:
        aggregate_cnv_samples
    output:
        done=os.path.join(OUTPUT_PATH, "results/cnv/{cell_line}/{split_key}/numbat_out/numbat_done.txt"),
    log:
        stdout="logs/numbat_aggregate_{cell_line}_{split_key}.stdout",
        stderr="logs/numbat_aggregate_{cell_line}_{split_key}.stderr",
    shell:
        """
        echo "All Numbat samples completed for {wildcards.cell_line}_{wildcards.split_key}" >> {log.stdout} 2>> {log.stderr}
        touch {output.done}
        """