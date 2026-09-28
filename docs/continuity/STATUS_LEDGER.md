## Phase 4C.1B.6R - Evidence Integrity, Statistical Contract and Repository Cleanup

- **Muc tieu**: Khép kín Phase 4C.1B.6 bằng evidence đo thực, sửa hợp đồng thống kê, dọn dẹp repo.
- **Starting commit**: 24adfce1 (Phase 4C.1B.6 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) EVAL-LEAK-001 fixed. (2) 7 regression tests. (3) AUROC CI: 0.6118 [0.568, 0.656]. (4) Dummy: 0.4117 (val-only). (5) Meta baseline: NOT_MEASURED. (6) Stage 2: INSUFFICIENT_EVIDENCE (meta NOT_MEASURED, single-seed). (7) Evidence at research/evidence/phase-4c.1b.6/.
- **Kiểm tra**: 145/145 Python pass, 70/70 TS, typecheck 0, build OK, continuity PASS. Archive SHA-256 MATCHES.
- **Kết quả**: Bundle ready for upload. Training runs = 0. Locked-test = 0. Colab = 0.
- **Evidence**: research/evidence/phase-4c.1b.6/ (10 files).
- **Quyết định tiếp theo**: Upload bundle, execute Colab N=50 seed=42.

---