import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType

@define
class LowerMedianRandomRecommender(RandomRecommender):
    """Random recommender restricted to candidates at/below the target's median.
    Defined for maximizing objective. When minimizing, the objective should be negated while creating.

    Reproduces the Gollum paper's initialization protocol: initial points are
    drawn uniformly at random, but only from the lower-median half of the
    (ground-truth) target distribution, instead of the full search space.
    """

    lookup: pd.DataFrame = field(kw_only=True)
    """Ground-truth table used to score candidates (must contain `target_name`)."""

    target_name: str = field(default="yield", kw_only=True)
    """Name of the target column in `lookup`."""

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        if searchspace.type != SearchSpaceType.DISCRETE:
            # no ground-truth-based restriction defined for continuous/hybrid parts
            return super()._recommend_hybrid(searchspace, candidates_exp, batch_size)

        param_cols = [p.name for p in searchspace.parameters]

        # one row per combo (avg over replicates) so the merge can't multiply rows
        lookup_scores = (
            self.lookup[param_cols + [self.target_name]]
            .groupby(param_cols, as_index=False)[self.target_name].mean()
        )
        scored = candidates_exp.merge(lookup_scores, on=param_cols, how="left")
        scored.index = candidates_exp.index

        median = self.lookup[self.target_name].median()
        lower_half = scored[scored[self.target_name] <= median]

        if lower_half.empty:
            raise ValueError(
                "No candidates found at or below the target median — check that "
                "`lookup` covers the search space's parameter combinations."
            )

        sampled_idx = lower_half.sample(
            n=batch_size, replace=len(lower_half) < batch_size
        ).index
        return candidates_exp.loc[sampled_idx]