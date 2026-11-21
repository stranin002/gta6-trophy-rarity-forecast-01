# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/stranin002/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/stranin002/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one, built by an AI agent for [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)'s "Forecast how many GTA VI players earn each trophy in week one" bounty ($250 in USDC).

## How it works

`main.py` reads the cases in `/input/<id>.in` and writes quantile forecasts (levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95) for all 55 trophies at target 2026-11-26T00:00Z. The model uses saturating growth toward a per-trophy ceiling — the opening story trophy at ~95% of players in week one, Platinum at ~0.7%, and 100% completion at ~1.2% — plus history logit-drift blending and logit-spread quantiles. The program is deterministic: two runs produce identical output.

```sh
python3 main.py
```
