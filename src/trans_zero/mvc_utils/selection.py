import torch as th

from .policies import PolicyDistribution
from .utility_functions import get_children_visits, get_transformed_default_values, policy_value


# use distributional selection policies instead of OptionalPolicy
class SelectionPolicy(PolicyDistribution):
    def __init__(self, *args, temperature: float = 0.0, **kwargs) -> None:
        # by default, we use argmax in selection
        super().__init__(*args, temperature=temperature, **kwargs)


class UCT(SelectionPolicy):
    def __init__(self, c: float, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def Q(self, node) -> th.Tensor:
        return get_transformed_default_values(node, self.value_transform)

    def _probs(self, node) -> th.Tensor:
        child_visits = get_children_visits(node)
        # if any child_visit is 0
        if th.any(child_visits == 0):
            # return 1 for all children with 0 visits
            return child_visits == 0

        return self.Q(node) + self.c * th.sqrt(th.log(th.tensor(node.visits)) / child_visits)



class PUCT(UCT):
    def _probs(self, node) -> th.Tensor:
        child_visits = get_children_visits(node)
        # if any child_visit is 0
        unvisited = child_visits == 0
        if th.any(unvisited):
            return node.prior_policy * unvisited

        return self.Q(node) + self.c * node.prior_policy * th.sqrt(th.tensor(node.visits)) / (child_visits + 1)

