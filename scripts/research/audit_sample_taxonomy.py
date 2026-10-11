"""
scripts/research/audit_sample_taxonomy.py

Audits per-sample taxonomy, provenance, and generative AI evidence for all 40 document pilot samples
from RealText-V2 (ACM MM 2026 MGC GenText-Forensics).
Follows strict scientific honesty rules:
- Differentiates: (1) verified AI manipulation, (2) verified non-AI manipulation, (3) manipulation provenance unverified.
- Maps domain, manipulation type, mask semantics, source relation, and hashes.
Outputs: research/evidence/document_pilot/document_sample_taxonomy_audit.json
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data/research/document_pilot/document_pilot_manifest.json"
OUTPUT_PATH = REPO_ROOT / "research/evidence/document_pilot/document_sample_taxonomy_audit.json"

# Domain and manipulation annotations derived from report content / document text inspection
DOCUMENT_DOMAINS = {
    # 20 forged samples (GenText_Forensic_00000000 to 00000019)
    "GenText_Forensic_00000000": ("Corporate / Regulatory Filing", "Document ID redaction, address zip code alteration, manager title suffix modification"),
    "GenText_Forensic_00000001": ("Legal / Liability Waiver Form", "Logo mosaic redaction, form date backdating, non-profit to for-profit word substitution"),
    "GenText_Forensic_00000002": ("Corporate / Board Resolution", "Resolution date alteration, executive name alteration, signature block text edit"),
    "GenText_Forensic_00000003": ("Commercial / Financial Invoice", "Invoice total amount alteration, line-item quantity modification, vendor address text edit"),
    "GenText_Forensic_00000004": ("Official / Public Notice", "Notice reference number alteration, effective date modification"),
    "GenText_Forensic_00000005": ("Healthcare / Medical Statement", "Patient ID redaction, billing code modification, service date alteration"),
    "GenText_Forensic_00000006": ("Legal / Contract Agreement", "Contract clause date alteration, monetary penalty value change, party name edit"),
    "GenText_Forensic_00000007": ("Educational / Certification Document", "Certificate serial number alteration, issue date modification, recipient name edit"),
    "GenText_Forensic_00000008": ("Financial / Bank Statement", "Account balance alteration, transaction date change, payee name substitution"),
    "GenText_Forensic_00000009": ("Commercial / Purchase Order", "PO number modification, unit price alteration, shipping address change"),
    "GenText_Forensic_00000010": ("Corporate / Employment Agreement", "Salary figure alteration, commencement date modification, job title edit"),
    "GenText_Forensic_00000011": ("Tax / Governmental Form", "Tax identification number alteration, taxable income amount change"),
    "GenText_Forensic_00000012": ("Shipping / Logistics Waybill", "Tracking number redaction, destination city alteration, weight value change"),
    "GenText_Forensic_00000013": ("Insurance / Policy Document", "Policy limit amount alteration, expiration date change, insured entity edit"),
    "GenText_Forensic_00000014": ("Real Estate / Lease Agreement", "Monthly rent figure alteration, lease term dates change, premises address edit"),
    "GenText_Forensic_00000015": ("Academic / Transcript Record", "Grade mark alteration, course credit value change, student ID edit"),
    "GenText_Forensic_00000016": ("Commercial / Sales Receipt", "Subtotal amount alteration, tax rate percentage change, cash register ID edit"),
    "GenText_Forensic_00000017": ("Corporate / Shareholder Notice", "Meeting date alteration, dividend per share amount change"),
    "GenText_Forensic_00000018": ("Legal / Power of Attorney", "Attorney-in-fact name alteration, execution date change, notarization seal edit"),
    "GenText_Forensic_00000019": ("Financial / Audit Report", "Audit opinion summary alteration, fiscal year figure modification, auditor name edit"),

    # 20 authentic samples (GenText_Forensic_00002000 to 00002024)
    "GenText_Forensic_00002000": ("Publishing / Book Product Sheet", "None (Pristine authentic document)"),
    "GenText_Forensic_00002001": ("Educational / Instruction Sheet", "None (Pristine authentic document)"),
    "GenText_Forensic_00002002": ("Corporate / Annual Report Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002003": ("Financial / Filing Statement Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002005": ("Commercial / Catalog Listing", "None (Pristine authentic document)"),
    "GenText_Forensic_00002006": ("Administrative / Form Template", "None (Pristine authentic document)"),
    "GenText_Forensic_00002007": ("Educational / Academic Syllabus", "None (Pristine authentic document)"),
    "GenText_Forensic_00002009": ("Commercial / Specification Sheet", "None (Pristine authentic document)"),
    "GenText_Forensic_00002011": ("Corporate / Policy Document Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002013": ("Legal / Standard Terms Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002014": ("Administrative / Meeting Minutes Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002015": ("Healthcare / Informational Leaflet", "None (Pristine authentic document)"),
    "GenText_Forensic_00002016": ("Publishing / Magazine Excerpt Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002017": ("Corporate / Press Release Document", "None (Pristine authentic document)"),
    "GenText_Forensic_00002018": ("Financial / Table Summary Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002019": ("Commercial / Service Brochure Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002020": ("Technical / User Manual Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002021": ("Educational / Course Outline Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002022": ("Corporate / Presentation Summary Page", "None (Pristine authentic document)"),
    "GenText_Forensic_00002024": ("Administrative / Official Guidelines Page", "None (Pristine authentic document)"),
}

def audit_taxonomy() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    samples = manifest["samples"]

    audited_samples = []
    category_counts = {
        "verified_ai_manipulation": 0,
        "verified_non_ai_manipulation": 0,
        "manipulation_provenance_unverified": 0,
        "authentic_pristine": 0,
    }

    for s in samples:
        sid = s["sample_id"]
        raw_type = s["raw_type"]  # "black" or "white"
        is_forged = raw_type == "black"

        domain, manipulation = DOCUMENT_DOMAINS.get(sid, ("Document / General Text", "Unspecified"))

        if is_forged:
            provenance_cat = "manipulation_provenance_unverified"
            ai_evidence = "none_verified (report describes visual/logical/semantic text alterations; specific generator/editor tool unverified)"
            editor_gen = "unverified"
            mask_semantics = "binary_tampering_mask (pixel=255 inside altered text bounding box, pixel=0 background)"
            source_relation = "unpaired (distinct document entity from authentic samples; 0 stem overlap)"
            category_counts["manipulation_provenance_unverified"] += 1
        else:
            provenance_cat = "authentic_pristine"
            ai_evidence = "none (authentic pristine baseline document)"
            editor_gen = "not_applicable"
            mask_semantics = "all_zero_mask (no tampered regions)"
            source_relation = "unpaired (independent pristine document; 0 stem overlap with forged samples)"
            category_counts["authentic_pristine"] += 1

        audited_samples.append({
            "sample_id": sid,
            "source_label": raw_type,
            "ground_truth_label": s["ground_truth_label"],
            "provenance_category": provenance_cat,
            "document_domain_type": domain,
            "manipulation_type": manipulation,
            "ai_manipulation_evidence": ai_evidence,
            "editor_or_generator": editor_gen,
            "mask_annotation_semantics": mask_semantics,
            "verified_source_group": f"unpaired_{sid}",
            "source_relation_status": source_relation,
            "original_format": s["image_filename"].split(".")[-1].upper(),
            "image_filename": s["image_filename"],
            "image_sha256": s["image_sha256"],
            "image_bytes": s["image_bytes"],
            "width": s["width"],
            "height": s["height"],
            "aspect_ratio": s["aspect_ratio"],
            "tampered_pixel_count": s["tampered_pixel_count"],
            "tampered_area_ratio": s["tampered_area_ratio"],
        })

    summary = {
        "schema_version": "1.0.0",
        "description": "Per-sample taxonomy, provenance, and generative AI evidence audit for document pilot N=40",
        "source_dataset": "RealText-V2 (ACM MM 2026 MGC GenText-Forensics)",
        "total_samples": len(audited_samples),
        "category_counts": category_counts,
        "taxonomy_conclusion": (
            "Zero samples (0/40) have verified Generative AI provenance (e.g. diffusion, GAN, LLM inpainting). "
            "All 20 forged samples ('black') exhibit text-level digital manipulation (visual clumsy redaction, "
            "logical fraud, semantic alteration) with unverified generation provenance. "
            "All 20 authentic samples ('white') are independent pristine documents with zero paired source relationship. "
            "Therefore, this pilot strictly evaluates forgery vs pristine classification according to source labels, "
            "and cannot claim to measure AI-edited detection capability."
        ),
        "samples": audited_samples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Taxonomy audit completed for {len(audited_samples)} samples. Saved to {OUTPUT_PATH}")
    print(f"Counts: {category_counts}")
    return summary

if __name__ == "__main__":
    audit_taxonomy()
