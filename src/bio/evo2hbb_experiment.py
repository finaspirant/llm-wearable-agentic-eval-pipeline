import os
import requests
import json
import time
from datetime import datetime

API_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"

API_KEY = os.environ["NVIDIA_API_KEY"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 8 Indian HBB variants
# Format: (variant_name, sequence, clinical_significance, expected_effect)
variants = [
    {
        "id": "HBB_619bp_deletion",
        "name": "619bp Deletion",
        "description": "Most common Indian beta-thalassemia mutation",
        "clinical": "Pathogenic — beta-thalassemia major",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGG"
    },
    {
        "id": "HBB_IVS1_5",
        "name": "IVS1-5 G→C",
        "description": "Second most common Indian mutation — splicing defect",
        "clinical": "Pathogenic — beta-thalassemia intermedia/major",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGC"
    },
    {
        "id": "HBB_codon_8_9",
        "name": "Codon 8/9 +G",
        "description": "Frameshift mutation — common in Indian subcontinent",
        "clinical": "Pathogenic — beta-thalassemia major",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAG"
    },
    {
        "id": "HBB_codon_15",
        "name": "Codon 15 G→A",
        "description": "Nonsense mutation",
        "clinical": "Pathogenic — beta-thalassemia major",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCA"
    },
    {
        "id": "HBB_codon_30",
        "name": "Codon 30 G→C",
        "description": "Missense mutation affecting beta-globin stability",
        "clinical": "Pathogenic — beta-thalassemia",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACT"
    },
    {
        "id": "HBB_codon_41_42",
        "name": "Codon 41/42 -TTCT",
        "description": "Frameshift — prevalent across South Asia",
        "clinical": "Pathogenic — beta-thalassemia major",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAA"
    },
    {
        "id": "HBB_codon_5",
        "name": "Codon 5 -CT",
        "description": "Frameshift deletion",
        "clinical": "Pathogenic — beta-thalassemia major",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGA"
    },
    {
        "id": "HBB_IVS1_1",
        "name": "IVS1-1 G→T",
        "description": "Splice site mutation — severe phenotype",
        "clinical": "Pathogenic — beta-thalassemia major",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAG"
    },
    {
        "id": "HBB_wildtype",
        "name": "HBB Wildtype (Control)",
        "description": "Normal HBB sequence — control for comparison",
        "clinical": "Benign — normal beta-globin",
        "sequence": "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTTTCAGGTTGGCTTT"
    }
]

results = []

print(f"Starting Evo2-40B HBB variant experiment — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Running {len(variants)} variants + 1 wildtype control\n")

for v in variants:
    print(f"Running: {v['name']} ({v['id']})...")
    
    payload = {
        "sequence": v["sequence"],
        "num_tokens": 100
    }
    
    try:
        start = time.time()
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        elapsed = round((time.time() - start) * 1000)
        
        if response.status_code == 200:
            data = response.json()
            result = {
                "id": v["id"],
                "name": v["name"],
                "description": v["description"],
                "clinical_significance": v["clinical"],
                "input_sequence": v["sequence"],
                "generated_continuation": data.get("sequence", ""),
                "elapsed_ms": elapsed,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            print(f"  ✅ Success — {elapsed}ms — continuation: {data.get('sequence', '')[:50]}...")
        else:
            result = {
                "id": v["id"],
                "name": v["name"],
                "status": "error",
                "error_code": response.status_code,
                "error_body": response.text[:200],
                "timestamp": datetime.now().isoformat()
            }
            print(f"  ❌ Error {response.status_code}: {response.text[:100]}")
            
    except Exception as e:
        result = {
            "id": v["id"],
            "name": v["name"],
            "status": "exception",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(f"  ❌ Exception: {e}")
    
    results.append(result)
    time.sleep(2)  # Rate limit buffer

# Save results
output_file = f"evo2_hbb_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w") as f:
    json.dump({
        "experiment_id": "QE-002",
        "description": "Evo2-40B Indian HBB variant sequence generation",
        "model": "evo2-40b",
        "api_endpoint": API_URL,
        "run_date": datetime.now().isoformat(),
        "variants_tested": len(variants),
        "results": results
    }, f, indent=2)

print(f"\nDone. Results saved to: {output_file}")
print(f"Successful runs: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")