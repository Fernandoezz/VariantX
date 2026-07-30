sample_vcf_content = """##fileformat=VCFv4.2
##INFO=<ID=DP,Number=1,Type=Integer,Description="Read depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
1\t11794419\t.\tG\tA\t.\tPASS\tDP=45\tGT\t1/1
17\t43094692\t.\tG\tA\t.\tPASS\tDP=38\tGT\t0/1
7\t117559592\t.\tA\tT\t.\tPASS\tDP=52\tGT\t0/1
"""

with open("data/sample_patient/sample_patient.vcf", "w") as f:
    f.write(sample_vcf_content)

print("Sample VCF created.")