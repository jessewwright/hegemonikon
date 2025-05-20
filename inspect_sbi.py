import sbi
import sbi.analysis
import inspect

print(f"SBI version: {sbi.__version__}")

print("\n--- Contents of sbi.analysis ---")
analysis_contents = dir(sbi.analysis)
print(analysis_contents)

print("\n--- Functions specifically in sbi.analysis (not submodules/private) ---")
_func_count = 0
for name in analysis_contents:
    if not name.startswith('_'):
        try:
            obj = getattr(sbi.analysis, name)
            if inspect.isfunction(obj) and obj.__module__.startswith('sbi.analysis'):
                print(f"Function: {name}")
                _func_count += 1
                try:
                    print(inspect.signature(obj))
                except (ValueError, TypeError):
                    print("  (Could not get signature)")
        except AttributeError:
            pass # Some attributes might not be gettable or are not functions
if _func_count == 0:
    print('No functions found directly in sbi.analysis that are part of its top-level or submodules like plot.')

print("\n--- Potential SBC-related functions (from all attributes in sbi.analysis) ---")
sbc_candidates = [name for name in analysis_contents if 'sbc' in name.lower() or 'rank' in name.lower()]
sbc_func_count = 0
if sbc_candidates:
    for name in sbc_candidates:
        try:
            obj = getattr(sbi.analysis, name)
            # We are interested if it's a function AND its module indicates it's part of sbi.analysis (e.g. sbi.analysis.plot)
            if inspect.isfunction(obj) and obj.__module__.startswith('sbi.analysis'):
                 print(f"Candidate: sbi.analysis.{name} (from module: {obj.__module__})")
                 sbc_func_count += 1
        except AttributeError:
            print(f"Could not getattr sbi.analysis.{name}")
    if sbc_func_count == 0:
        print('No functions matching SBC/rank criteria found in sbi.analysis modules.')
else:
    print("No attributes containing 'sbc' or 'rank' found in sbi.analysis by simple name match.")

# Explicitly check for get_sbc_ranks and sbc_rank_stats if they weren't listed
# because they might be imported into sbi.analysis's namespace without being defined in a submodule Python sees as sbi.analysis.X
print("\n--- Explicit checks for get_sbc_ranks and sbc_rank_stats ---")
try:
    from sbi.analysis import get_sbc_ranks
    print("Successfully imported: from sbi.analysis import get_sbc_ranks")
    print(inspect.signature(get_sbc_ranks))
except ImportError:
    print("Failed to import: from sbi.analysis import get_sbc_ranks")
except Exception as e:
    print(f"Error inspecting get_sbc_ranks: {e}")

try:
    from sbi.analysis import sbc_rank_stats
    print("Successfully imported: from sbi.analysis import sbc_rank_stats")
    print(inspect.signature(sbc_rank_stats))
except ImportError:
    print("Failed to import: from sbi.analysis import sbc_rank_stats")
except Exception as e:
    print(f"Error inspecting sbc_rank_stats: {e}")
