# Dashboard Parameter Reference

This maps the parameters in `offshore_dashboard_exc.html` to the values and descriptions in the paper. The exercise's initial values are intentionally scrambled; they are shown here so it is clear which values are prompts and which values are supported by the paper. "Reference code" means the accompanying implementation in [MainOffshoreWave.py](MainOffshoreWave.py), not a value stated in the paper.

## Wave Climate and Load

| Dashboard parameter | Exercise starting value | Paper value / location | Direct quote from paper |
|---|---:|---|---|
| Simulation start year (`baseYear`) | 2012 | 2015; Section 4.1, Figure 3b, p. 6 ([text](energies-16-04771-v2.txt#L294)). | “The synthetic wave data for a period of 60 years, starting in 2015” |
| Simulation duration (`years`) | 53 years | 60 years; Section 4.1, p. 6 ([text](energies-16-04771-v2.txt#L251-L261), [Figure 3b description](energies-16-04771-v2.txt#L294)). | “over a large time span of 60 years” |
| Random seed (`seed`, hidden) | 0 | No fixed seed specified; the paper recommends repeated simulations, p. 9 ([text](energies-16-04771-v2.txt#L368-L376)). | “Repeated simulations can improve the results and the interpretation.” |
| Stormy months (`severe`) | Nov-Apr (first option) | November-March; Section 4.1, pp. 5-6 ([text](energies-16-04771-v2.txt#L245-L254)). | “November, December, January, February, and March are assumed to be severe weather months” |
| Stormy-month wave scale (`rs1`) | 3.1 m | Sigma range 0.7-1.5 by season; Fig. 3 uses average sigma 1.1, pp. 5-7 ([text](energies-16-04771-v2.txt#L245-L254), [caption](energies-16-04771-v2.txt#L301-L305)). Reference code uses 1.5. | “The scale of the Rayleigh distribution varies between σ = 0.7 and σ = 1.5” |
| Calm-month wave scale (`rs2`) | 0.9 m | Sigma range 0.7-1.5 by season; the paper does not assign an exact calm-month value. Reference code uses 0.7 ([text](energies-16-04771-v2.txt#L245-L254)). | “depending on the season” |
| Waves per month (`loadK`, displayed in thousands) | 1,734 | `Ntotal = 2.592 * 10^5` waves/month; Eq. 4, p. 6 ([text](energies-16-04771-v2.txt#L264-L284)). This is 259.2 in thousands; reference code rounds to 259. | “Ntotal = 2.592 · 10^5 per month” |
| Smallest counted wave (`Hmin`, load calculation) | 11 m | `Hmin = 8 m`; Eq. 3, p. 6 ([text](energies-16-04771-v2.txt#L275-L284)). | “with Hmin = 8 m and Hrel = 17 m” |

## Fatigue Limit

| Dashboard parameter | Exercise starting value | Paper value / location | Direct quote from paper |
|---|---:|---|---|
| Old structure: smallest counted wave (`minH_f`) | 5 m | `Hmin = 8 m`; Eq. 3, p. 6 ([text](energies-16-04771-v2.txt#L275-L281)). | “with Hmin = 8 m” |
| Old structure: strength (`maxH_f`) | 23 m | `Hrel = 17 m`; Eq. 3, p. 6 ([text](energies-16-04771-v2.txt#L275-L281)). | “Hrel = 17 m representing the 100-year wave height” |
| New structure: smallest counted wave (`minHDe_f`) | 12 m | No numerical value stated; separate new-structure curve in Figure 3c, p. 7 ([caption](energies-16-04771-v2.txt#L301-L310)). Reference code uses 10 m ([MainOffshoreWave.py](MainOffshoreWave.py#L277-L284)). | “For a variety of steel structures, different cyclic structural resistance curves would apply.” |
| New structure: strength (`maxHDe_f`) | 25 m | No numerical value stated; separate new-structure curve in Figure 3c, p. 7 ([caption](energies-16-04771-v2.txt#L301-L310)). Reference code uses 19 m ([MainOffshoreWave.py](MainOffshoreWave.py#L277-L284)). | “two types of steel structures are considered” |

## End-of-Life and Network

| Dashboard parameter | Exercise starting value | Paper value / location | Direct quote from paper |
|---|---:|---|---|
| Operation mode (`mode`, `eolMode`, `energyMode`) | Repowering (the linked two-option fields show Keep) | The paper compares extension, re-powering, and decommissioning; Sections 4.2 and Figures 5-7, pp. 7-9 ([text](energies-16-04771-v2.txt#L315-L366)). | “three options: extended operation, decommissioning, or re-powering” |
| End-of-life year (`eolYear`) | 2044 | A 25-year permit after a 2015 start implies 2040; paper permit example p. 2 and simulated series p. 6 ([permit](energies-16-04771-v2.txt#L68-L73), [simulation period](energies-16-04771-v2.txt#L294-L299)). Reference code uses 2040. | “The permit shall expire 25 years after commissioning” |
| Decommissioning finished (`decomYear`) | 2048 | No exact year stated; reference code uses 2042 ([MainOffshoreWave.py](MainOffshoreWave.py#L218-L221)). | “decommissioning and rebuilding take a certain amount of time” |
| Rebuild finished (`rebuildYear`) | 2053 | No exact year stated; reference code uses 2050 ([MainOffshoreWave.py](MainOffshoreWave.py#L218-L221)). | “a certain downtime was considered between the decommissioning of the old structure” |
| Number of substations (`numOSS`) | 2 | Figure 4 shows three OWFs and three OSSs, p. 7 ([text](energies-16-04771-v2.txt#L315-L321)); reference code uses three OSSs ([MainOffshoreWave.py](MainOffshoreWave.py#L329-L345)). | “OSS (grey from 31 to 33)” |
| Turbines per substation (`numT`) | 85 | 60 turbines per OWF in simulations; Section 4.2, p. 7 ([text](energies-16-04771-v2.txt#L315-L321)). Figure 4 shows 10 only for visualization. | “in the simulations, 60 turbines are employed per OWF” |
| Failure probability below limit (`plow`) | 0.35 | `p = 0.1` before the structural resistance is exceeded; Section 4.2, p. 8 ([text](energies-16-04771-v2.txt#L327-L335)). | “the failure probability after EOL remains low, with p = 0.1” |
| Failure probability after limit (`phigh`) | 0.8 | `p = 0.5` after the structural resistance is exceeded; Section 4.2, p. 8 ([text](energies-16-04771-v2.txt#L327-L335)). | “the failure probability increased to p = 0.5” |
| Damage wave threshold (`damage`, fixed) | 12 m | No `Hmax` damage threshold specified in the paper; reference code uses `Hmax > 15` ([MainOffshoreWave.py](MainOffshoreWave.py#L354-L362)). | [No corresponding threshold sentence in the paper.] |

## Energy Output and Other Model Constants

| Dashboard parameter | Exercise starting value | Paper value / location | Direct quote from paper |
|---|---:|---|---|
| Minimum/maximum turbine output (`minOut`, `maxOut`) | 3 / 11.5 MW | Baseline 4-6 MW; re-powering average 7.5 MW; decommissioning average 8.5 MW; Sections 4.2-4.3, pp. 7-9 ([baseline](energies-16-04771-v2.txt#L321-L326), [re-powering](energies-16-04771-v2.txt#L343-L350), [decommissioning](energies-16-04771-v2.txt#L353-L359)). Reference code ramps values over time ([MainOffshoreWave.py](MainOffshoreWave.py#L374-L414)). | “an energy output between 4 and 6 MW per turbine”; “7.5 MW per turbine, on average”; “8.5 MW per turbine, on average” |
| Maximum-wave extra-height cap (`maxH`, fixed at 20 m) | 20 m | Equation 2 gives the upper-bound formula but no 20 m cap ([text](energies-16-04771-v2.txt#L254-L261)); cap is implementation-specific. | “u = Hsig + Hsig · 0.07 · (y − ybase + 3)” |
| Repair duration | 2 months | Two months; Section 4.2, p. 8 ([text](energies-16-04771-v2.txt#L331-L339)). | “The duration of all needed repairs is set to two months” |

The exercise starting values above are not suggested answers. For values not numerically provided by the paper, the table identifies the corresponding figure or marks them as reference-code assumptions instead of treating them as paper-backed defaults.
