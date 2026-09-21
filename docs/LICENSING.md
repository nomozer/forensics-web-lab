# Licensing Decisions & Considerations: Forensics Web Lab

## 1. Current Repository Licensing Status

* **Status**: **Unlicensed / Proprietary (Pending Maintainer Decision)**.
* **Directive**: In strict accordance with project governance rules, no open-source or commercial license file (e.g. `LICENSE`, `LICENSE.md`) has been created for the root repository. Formal licensing is deferred to the repository owner / project leads.

---

## 2. Key Decisions for Repository Owners

When determining the open release or distribution terms for Forensics Web Lab, the project leads should consider the following options and tradeoffs:

### Option A: Permissive Open-Source (Apache 2.0 or MIT)
* **Pros**: Maximizes academic reach, community contributions, and adoption by journalists, researchers, and public interest non-profits.
* **Considerations**: Apache 2.0 provides an explicit patent grant and trademark restrictions, offering strong legal protection for academic research tools.

### Option B: Copyleft Open-Source (GNU GPLv3 or AGPLv3)
* **Pros**: Guarantees that any derivative tools, proprietary forks, or SaaS modifications must remain openly accessible with source code.
* **Considerations**: Imposes reciprocal obligations that may restrict commercial enterprise integrations.

### Option C: Dual Licensing (Academic Free / Commercial Enterprise)
* **Pros**: Free for verified non-commercial research, academic institutions, and investigative journalism, with commercial usage requiring separate licensing.
* **Considerations**: Requires governance overhead to manage commercial inquiries and compliance.

---

## 3. Third-Party Dependency & Dataset License Considerations

1. **Client-Side Web Dependencies**:
   * All chosen npm libraries (React, Vite, `onnxruntime-web`) are distributed under standard permissive licenses (MIT, Apache 2.0).
2. **Machine Learning Datasets**:
   * Academic datasets (e.g. `GenImage`, `RealHD`) are published under non-commercial licenses (`CC-BY-NC 4.0`).
   * Models trained exclusively on non-commercial academic datasets cannot be licensed or deployed for direct commercial monetization without acquiring commercial rights to underlying training data.
