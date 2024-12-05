# License: GPTQModel/licenses/LICENSE.mit
from typing import List

from device_smi import Device
from thefuzz import process
from tvm.target import Target
from tvm.target.tag import list_tags

from ...utils.logger import setup_logger

logger = setup_logger()

TARGET_MISSING_ERROR = (
    "TVM target not found. Please set the TVM target environment variable using `export TVM_TARGET=<target>`, "
    "where <target> is one of the available targets can be found in the output of `tools/get_available_targets.py`."
)

def find_best_match(tags, query):
    """
    Finds the best match for a query within a list of tags using fuzzy string matching.
    """
    MATCH_THRESHOLD = 25
    best_match, score = process.extractOne(query, tags)
    # print(f"TVM arch find_best_match: best_match = {best_match}, score = {score}")

    def check_target(best, default):
        # print(f"Target(best) = {Target(best)}, Target(default)  = {Target(default)}")
        return best if Target(best).arch == Target(default).arch else default

    if check_target(best_match, "cuda") == best_match:
        match = best_match if score >= MATCH_THRESHOLD else "cuda"
        logger.info(f"found best match: {match}")
        return match
    else:
        logger.warning(TARGET_MISSING_ERROR)
        return "cuda"


def get_all_nvidia_targets() -> List[str]:
    """
    Returns all available NVIDIA targets.
    """
    all_tags = list_tags()
    return [tag for tag in all_tags if "nvidia" in tag]


def patched_auto_detect_nvidia_target(gpu_id: int = 0) -> str:
    """
    Automatically detects the NVIDIA GPU architecture to set the appropriate TVM target.

    Returns:
        str: The detected TVM target architecture.
    """
    # Return a predefined target if specified in the environment variable
    # if "TVM_TARGET" in os.environ:
    #     return os.environ["TVM_TARGET"]

    # Fetch all available tags and filter for NVIDIA tags
    all_tags = list_tags()
    nvidia_tags = [tag for tag in all_tags if "nvidia" in tag]

    # Get the current GPU model and find the best matching target
    gpu_model = Device(f"cuda:{gpu_id}").model
    # print(f"gpu_model: {gpu_model}")

    # compat: Nvidia makes several oem (non-public) versions of A100 and perhaps other models that
    # do not have clearly defined TVM matching target so we need to manually map them to the correct one.
    if "pg506-230" in gpu_model or "pg506-232" in gpu_model.lower():
        gpu_model = "NVIDIA A100"

    # print("GPU_model",gpu_model)
    target = find_best_match(nvidia_tags, gpu_model) if gpu_model else "cuda"
    return target
