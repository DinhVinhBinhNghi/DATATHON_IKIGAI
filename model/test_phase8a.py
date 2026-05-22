import importlib.util
from pathlib import Path

scripts = [
    "scripts/00_check_data.py",
    "scripts/10_preaggregate.py",
    "scripts/11_build_candidates_train.py",
    "scripts/12_build_candidates_predict.py",
    "scripts/13_train_ranker.py",
    "scripts/14_score_candidates.py",
    "scripts/15_rerank_and_submit.py",
    "scripts/16_evaluate_candidates.py",
    "scripts/17_evaluate_local.py",
    "scripts/18_analyze_marketplace_health.py",
]
all_ok = True
for s in scripts:
    spec = importlib.util.spec_from_file_location(Path(s).stem, s)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print(f"  OK  {s}")
    except SystemExit:
        # main() sys.exit, vì module này không có if __name__... bypass nên đôi khi
        # script chạy luôn. Chấp nhận, miễn không có ImportError.
        print(f"  OK  {s} (exited)")
    except Exception as e:
        print(f"  FAIL {s}: {e}")
        all_ok = False

if all_ok:
    print("Phase 8A: All 10 scripts import OK")
else:
    print("Phase 8A: Some scripts failed")
