from sotopia_rl.prompter.all_the_same_attribution_function import (
    get_attribution_single_conv as all_the_same_attribution_normalized_single_conv,
)
from sotopia_rl.prompter.direct_attribution_function import (
    get_attribution_single_conv as direct_attribution_single_conv,
)
from sotopia_rl.prompter.direct_attribution_generic_function import (
    get_attribution_single_conv as direct_attribution_generic_single_conv,
)
from sotopia_rl.prompter.direct_attribution_normalized_function import (
    get_attribution_single_conv as direct_attribution_normalized_single_conv,
)
from sotopia_rl.prompter.discounting_attribution_function import (
    get_attribution_single_conv as discounting_attribution_single_conv,
)
from sotopia_rl.prompter.goal_progress_attribution_function import (
    get_attribution_single_conv as goal_progress_attribution_single_conv,
)
from sotopia_rl.prompter.key_utterance_function import (
    get_attribution_single_conv as key_utterance_attribution_single_conv,
)
from sotopia_rl.prompter.only_response_attribution_function import (
    get_attribution_single_conv as only_response_attribution_single_conv,
)
from sotopia_rl.prompter.utterance_quality_attribution_function import (
    get_attribution_single_conv as utterance_quality_attribution_single_conv,
)
from sotopia_rl.prompter.utterance_quality_attribution_normalized_function import (
    get_attribution_single_conv as utterance_quality_attribution_normalized_single_conv,
)

# from sotopia_rl.prompter.utterance_quality_generic_function import (
#     get_attribution_single_conv as utterance_quality_attribution_generic_single_conv,
# )

ATTRIBUTION_METHOD_DICT = {
    "discounting": discounting_attribution_single_conv,
    "direct": direct_attribution_single_conv,
    "direct_normalized": direct_attribution_normalized_single_conv,
    "goal_progress": goal_progress_attribution_single_conv,
    "only_response": only_response_attribution_single_conv,
    "utterance_quality": utterance_quality_attribution_single_conv,
    "utterance_quality_normalized": utterance_quality_attribution_normalized_single_conv,
    "all_the_same": all_the_same_attribution_normalized_single_conv,
    "key_utterance": key_utterance_attribution_single_conv,
    "direct_generic": direct_attribution_generic_single_conv,
    # "utterance_quality_generic": utterance_quality_attribution_generic_single_conv,
}
