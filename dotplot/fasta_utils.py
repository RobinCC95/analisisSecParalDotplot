from Bio import SeqIO

def cargar_secuencias_fasta(file1, file2):
    def merge_sequences(file_path):
        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))
        return "".join(sequences)

    secuencia1 = merge_sequences(file1)
    secuencia2 = merge_sequences(file2)
    return secuencia1, secuencia2