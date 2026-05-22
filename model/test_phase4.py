from src.candidates import run_build_candidates
from src.candidates.sources.reengagement import build_reengagement
from src.candidates.sources.covisit import build_covisit_matrix
from src.candidates.sources.category_pop import build_category_pop_candidates
from src.candidates.sources.city_cat_pop import build_city_cat_pop_candidates
from src.candidates.sources.global_pop import build_global_pop_candidates
from src.candidates.merge import merge_candidates
print("Phase 4 import OK")
print("Functions:")
for f in [run_build_candidates, build_reengagement, build_covisit_matrix,
          build_category_pop_candidates, build_city_cat_pop_candidates,
          build_global_pop_candidates, merge_candidates]:
    print(f"  {f.__name__}")
