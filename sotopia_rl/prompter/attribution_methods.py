from sotopia_rl.prompter.direct_attribution_function import (
    get_attribution_single_conv as direct_attribution_single_conv,
)
from sotopia_rl.prompter.direct_average_attribution_function import (
    get_attribution_single_conv as direct_average_attribution_single_conv,
)
from sotopia_rl.prompter.discounting_attribution_function import (
    get_attribution_single_conv as discounting_attribution_single_conv,
)
from sotopia_rl.prompter.goal_progress_attribution_function import (
    get_attribution_single_conv as goal_progress_attribution_single_conv,
)
from sotopia_rl.prompter.only_response_attribution_function import (
    get_attribution_single_conv as only_response_attribution_single_conv,
)

ATTRIBUTION_METHOD_DICT = {
    "discounting": discounting_attribution_single_conv,
    "direct": direct_attribution_single_conv,
    "direct_average": direct_average_attribution_single_conv,
    "goal_progress": goal_progress_attribution_single_conv,
    "only_response": only_response_attribution_single_conv
}
