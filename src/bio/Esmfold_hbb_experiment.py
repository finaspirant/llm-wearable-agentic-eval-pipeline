import requests
import json
import time
import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))

# NVIDIA NIM ESMFold endpoint (no /predict suffix — returns {"pdbs": [...]})
ESMFOLD_URL = "https://health.api.nvidia.com/v1/biology/meta/esmfold"

API_KEY = os.environ["NVIDIA_API_KEY"]


headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Codon table for DNA → protein translation
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

def mean_plddt_from_pdb(pdb_str: str) -> float:
    """Extract mean pLDDT from ESMFold PDB B-factor column (cols 61-66)."""
    scores = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM"):
            try:
                scores.append(float(line[60:66].strip()))
            except ValueError:
                pass
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def dna_to_protein(dna_seq):
    """Translate DNA sequence to amino acid sequence, stop at first stop codon."""
    dna_seq = dna_seq.upper().replace(' ', '').replace('\n', '')
    protein = []
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i+3]
        if len(codon) < 3:
            break
        aa = CODON_TABLE.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)

# Load Evo2 results to get generated continuations
# Update this path to match your actual output file
EVO2_RESULTS_FILE = os.path.join(_HERE, "evo2_hbb_results_20260629_213658.json")

try:
    with open(EVO2_RESULTS_FILE) as f:
        evo2_data = json.load(f)
    print(f"Loaded Evo2 results: {len(evo2_data['results'])} variants")
except FileNotFoundError:
    print(f"ERROR: Could not find {EVO2_RESULTS_FILE}")
    print("Please update EVO2_RESULTS_FILE path to match your output file")
    exit(1)

results = []

print(f"\nStarting ESMFold HBB structure prediction — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Translating Evo2 DNA continuations → amino acids → 3D structure\n")

for variant in evo2_data['results']:
    if variant['status'] != 'success':
        print(f"Skipping {variant['name']} — Evo2 run failed")
        continue

    variant_id = variant['id']
    variant_name = variant['name']

    # Combine input sequence + Evo2 continuation for full sequence
    full_dna = variant['input_sequence'] + variant.get('generated_continuation', '')
    
    # Translate to protein
    protein_seq = dna_to_protein(full_dna)
    
    # ESMFold max is 768 residues — truncate if needed
    if len(protein_seq) > 768:
        protein_seq = protein_seq[:768]
        truncated = True
    else:
        truncated = False

    if len(protein_seq) < 10:
        print(f"  ⚠️  {variant_name}: protein sequence too short ({len(protein_seq)} aa) — skipping")
        continue

    print(f"Running: {variant_name}")
    print(f"  DNA length: {len(full_dna)}bp → Protein: {len(protein_seq)} aa{' (truncated)' if truncated else ''}")

    payload = {
        "sequence": protein_seq
    }

    try:
        start = time.time()
        response = requests.post(ESMFOLD_URL, headers=headers, json=payload, timeout=120)
        elapsed = round((time.time() - start) * 1000)

        if response.status_code == 200:
            data = response.json()

            # API returns {"pdbs": ["ATOM ..."]}, pLDDT in B-factor column
            pdb_str = data.get('pdbs', [''])[0]
            plddt = mean_plddt_from_pdb(pdb_str)

            result = {
                "id": variant_id,
                "name": variant_name,
                "clinical_significance": variant.get('clinical_significance', ''),
                "dna_length_bp": len(full_dna),
                "protein_length_aa": len(protein_seq),
                "protein_sequence": protein_seq,
                "truncated": truncated,
                "mean_plddt": plddt,
                "pdb_structure": pdb_str[:200] + "...[truncated for JSON]" if pdb_str else '',
                "elapsed_ms": elapsed,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            print(f"  ✅ Success — {elapsed}ms | pLDDT: {plddt}")

        else:
            result = {
                "id": variant_id,
                "name": variant_name,
                "status": "error",
                "error_code": response.status_code,
                "error_body": response.text[:300],
                "timestamp": datetime.now().isoformat()
            }
            print(f"  ❌ Error {response.status_code}: {response.text[:150]}")

    except Exception as e:
        result = {
            "id": variant_id,
            "name": variant_name,
            "status": "exception",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(f"  ❌ Exception: {e}")

    results.append(result)
    time.sleep(3)  # ESMFold is heavier — slightly longer rate limit buffer

# Save results
output_file = os.path.join(_HERE, f"esmfold_hbb_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(output_file, "w") as f:
    json.dump({
        "experiment_id": "QE-003",
        "description": "ESMFold structure prediction on Evo2-generated HBB variant continuations",
        "model": "esmfold",
        "api_endpoint": ESMFOLD_URL,
        "upstream_experiment": "QE-002",
        "run_date": datetime.now().isoformat(),
        "variants_tested": len(results),
        "results": results
    }, f, indent=2)

print(f"\nDone. Results saved to: {output_file}")
print(f"Successful: {sum(1 for r in results if r.get('status') == 'success')}/{len(results)}")
print("\nKey metrics to look for:")
print("  pLDDT > 70 = confident structure prediction")
print("  pLDDT < 50 = disordered/uncertain region")
print("  Compare wildtype pLDDT vs pathogenic variants — divergence is your signal")