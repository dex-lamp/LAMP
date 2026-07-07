# Media Guide

The project page at https://dex-lamp.github.io/ is the best place for videos
and dense result figures. Keep the code repository focused on orientation and
usage.

## Recommended Assets

| Use case | Recommended asset | Rationale |
| --- | --- | --- |
| Repository README | `assets/main-pipeline.png` | Explains how the released modules connect: prior pretraining, BC, and residual RL. |
| Project-page hero or paper teaser | `abstract-render.png` from the project page | Strong visual summary with task photos and headline results. |
| Results section | `success-failure-summary.png` from the project page | Best for quantitative comparison after readers understand the method. |
| Detailed ablation discussion | `ablation-failure-breakdown.png` and `action-smoothness-curves.png` | Useful in paper/project-page results, too dense for the README top. |
| Videos | Project-page rollout grid | The page already hosts 48 IL/RL rollout videos without bloating the code repo. |

## Current Repository Choice

The root README uses `assets/main-pipeline.png`. I would keep rollout videos on
the project page and avoid committing `.mp4` files to this repository unless a
small demo clip becomes essential for documentation.
