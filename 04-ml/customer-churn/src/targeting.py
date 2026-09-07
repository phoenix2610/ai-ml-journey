"""Turning churn probabilities into a retention campaign that pays for itself.

The step most churn projects skip. A ranked risk list is not a decision — you
still have to choose *how many* customers to contact, and that choice has an
optimum you can compute.

The arithmetic per contacted customer:

    expected_gain = P(churn) × save_rate × customer_value − contact_cost

Two things make the naive version wrong:

**Not everyone you save was going to leave.** Only `P(churn)` of them were at
risk at all, and of those you only save `save_rate`. Contacting a customer with
a 5% churn probability is almost pure cost.

**Some customers are worth more than others.** Value is `monthly_charges ×
expected remaining months`, so a high-risk cheap customer can be worth less than
a medium-risk expensive one. Ranking by probability alone gets this wrong;
ranking by *expected value* is what the campaign should use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Outreach is cheap; the retention *offer* is not. A saved customer typically
# costs a real discount or credit, and pricing only the phone call is the most
# common way a churn model "proves" you should contact everybody.
CONTACT_COST = 45.0         # staff time + the incentive actually given
SAVE_RATE = 0.25            # fraction of genuinely at-risk customers retained
HORIZON_MONTHS = 12         # how far ahead value is counted


def customer_value(
    monthly_charges: pd.Series, *, horizon: int = HORIZON_MONTHS
) -> pd.Series:
    """Simple value model: what we lose over the horizon if they leave."""
    return monthly_charges.astype(float) * horizon


def expected_gain(
    probability: pd.Series,
    value: pd.Series,
    *,
    contact_cost: float = CONTACT_COST,
    save_rate: float = SAVE_RATE,
) -> pd.Series:
    """Expected profit from contacting each customer. May be negative."""
    return probability.astype(float) * save_rate * value.astype(float) - contact_cost


def build_campaign(
    probability: pd.Series,
    monthly_charges: pd.Series,
    *,
    contact_cost: float = CONTACT_COST,
    save_rate: float = SAVE_RATE,
    horizon: int = HORIZON_MONTHS,
    ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Rank every customer by expected gain -- not by probability."""
    value = customer_value(monthly_charges, horizon=horizon)
    gain = expected_gain(
        probability, value, contact_cost=contact_cost, save_rate=save_rate
    )

    frame = pd.DataFrame(
        {
            "churn_probability": probability.astype(float).values,
            "customer_value": value.values,
            "expected_gain": gain.values,
        },
        index=probability.index,
    )
    if ids is not None:
        frame.insert(0, "customer_id", ids.values)

    frame = frame.sort_values("expected_gain", ascending=False)
    frame["cumulative_gain"] = frame["expected_gain"].cumsum()
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


@dataclass
class CampaignPlan:
    contact_count: int
    total_customers: int
    expected_profit: float
    contact_cost: float
    expected_saves: float
    cutoff_probability: float

    @property
    def contact_share(self) -> float:
        return 100 * self.contact_count / self.total_customers if self.total_customers else 0.0

    def __str__(self) -> str:
        return (
            f"    contact          {self.contact_count:,} of {self.total_customers:,} "
            f"({self.contact_share:.1f}%)\n"
            f"    campaign cost    {self.contact_cost:>12,.0f}\n"
            f"    expected saves   {self.expected_saves:>12,.1f} customers\n"
            f"    expected profit  {self.expected_profit:>12,.0f}\n"
            f"    cutoff P(churn)  {self.cutoff_probability:>12.3f}"
        )


def optimal_campaign(
    campaign: pd.DataFrame,
    *,
    contact_cost: float = CONTACT_COST,
    save_rate: float = SAVE_RATE,
) -> CampaignPlan:
    """Contact everyone with a positive expected gain, and nobody else.

    Because the frame is sorted by expected gain, the optimum is simply the
    prefix where gain is still positive -- cumulative profit peaks exactly
    where individual gain crosses zero.
    """
    profitable = campaign[campaign["expected_gain"] > 0]
    count = len(profitable)

    return CampaignPlan(
        contact_count=count,
        total_customers=len(campaign),
        expected_profit=float(profitable["expected_gain"].sum()),
        contact_cost=count * contact_cost,
        expected_saves=float(profitable["churn_probability"].sum() * save_rate),
        cutoff_probability=float(profitable["churn_probability"].min()) if count else 0.0,
    )


def campaign_at_budget(
    campaign: pd.DataFrame, budget: float, *, contact_cost: float = CONTACT_COST,
    save_rate: float = SAVE_RATE,
) -> CampaignPlan:
    """Best campaign under a fixed budget -- take the top N you can afford."""
    affordable = int(budget // contact_cost)
    selected = campaign.head(affordable)

    return CampaignPlan(
        contact_count=len(selected),
        total_customers=len(campaign),
        expected_profit=float(selected["expected_gain"].sum()),
        contact_cost=len(selected) * contact_cost,
        expected_saves=float(selected["churn_probability"].sum() * save_rate),
        cutoff_probability=float(selected["churn_probability"].min()) if len(selected) else 0.0,
    )


def compare_to_probability_ranking(
    campaign: pd.DataFrame, n_contacts: int
) -> dict[str, float]:
    """What ranking by expected value buys over ranking by probability alone."""
    by_value = campaign.head(n_contacts)["expected_gain"].sum()
    by_probability = (
        campaign.sort_values("churn_probability", ascending=False)
        .head(n_contacts)["expected_gain"]
        .sum()
    )
    return {
        "by_expected_value": float(by_value),
        "by_probability": float(by_probability),
        "uplift": float(by_value - by_probability),
    }


__all__ = [
    "customer_value", "expected_gain", "build_campaign", "optimal_campaign",
    "campaign_at_budget", "compare_to_probability_ranking",
    "CampaignPlan", "CONTACT_COST", "SAVE_RATE", "HORIZON_MONTHS",
]
