"""Import smoke check for the public residual RL package."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reinforcement_learning.residual_rl.agents.continuous.residual_sac import (
    ResidualSACAgent,
)
from reinforcement_learning.residual_rl.agents.continuous.sac import SACAgent
from reinforcement_learning.residual_rl.networks.bc_vae import BCVAEPolicy
from reinforcement_learning.residual_rl.utils.bc_vae import BCVAECheckpoint
from reinforcement_learning.residual_rl.utils.factory import make_residual_sac_pixel_agent
from reinforcement_learning.residual_rl.wrappers.bc_vae_context import BCVAEPolicyContext


def main() -> None:
    symbols = [
        ResidualSACAgent,
        SACAgent,
        BCVAEPolicy,
        BCVAECheckpoint,
        BCVAEPolicyContext,
        make_residual_sac_pixel_agent,
    ]
    print("residual_rl import smoke check passed:", ", ".join(s.__name__ for s in symbols))


if __name__ == "__main__":
    main()
