sample_cnv_vcf_content = """##fileformat=VCFv4.2
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
17\t43044295\t.\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=43125483\tGT\t0/1
7\t117480025\t.\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=117485000\tGT\t0/1
"""

with open("data/sample_patient/sample_patient_cnv.vcf", "w") as f:
    f.write(sample_cnv_vcf_content)

print("Sample CNV VCF created.")