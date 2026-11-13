# Dashboard Parameter Reference

This is a paper-only lookup for the parameters in `offshore_dashboard_exc.html`. It omits the exercise's randomized starting values. Where the paper does not give a numerical value, the location column points to the closest relevant passage or figure and says so explicitly.

## Wave Climate and Load

| Dashboard parameter | Paper value / where to look | Paper quotation |
|---|---|---|
| Simulation start year (`baseYear`) | 2015; Section 4.1, Figure 3b, p. 6 | “The synthetic wave data for a period of 60 years, starting in 2015” |
| Simulation duration (`years`) | 60 years; Section 4.1, p. 6; Figure 3b | “over a large time span of 60 years” |
| Random seed (`seed`, hidden) | No fixed seed stated; discussion of randomness and repeated simulations, p. 9 | “Repeated simulations can improve the results and the interpretation.” |
| Stormy months (`severe`) | November-March; Section 4.1, pp. 5-6 | “November, December, January, February, and March are assumed to be severe weather months” |
| Stormy-month wave scale (`rs1`) | `σ = 1.5`, upper end of the seasonal range; Section 4.1, pp. 5-6. Figure 3 uses average `σ = 1.1`, p. 7. | “severe weather months with large σ” |
| Calm-month wave scale (`rs2`) | `σ = 0.7`, lower end of the seasonal range; Section 4.1, pp. 5-6 | “varies between σ = 0.7 and σ = 1.5, depending on the season” |
| Waves per month (`loadK`, displayed in thousands) | `Ntotal = 2.592 * 10^5` waves/month (259.2 thousand); Eq. 4, p. 6 | “Ntotal = 2.592 · 10^5 per month” |
| Smallest counted wave (`Hmin`, load calculation) | `Hmin = 8 m`; Eq. 3, p. 6 | “with Hmin = 8 m and Hrel = 17 m” |

## Fatigue Limit

| Dashboard parameter | Paper value / where to look | Paper quotation |
|---|---|---|
| Old structure: smallest counted wave (`minH_f`) | `Hmin = 8 m`; Eq. 3, p. 6 | “with Hmin = 8 m” |
| Old structure: strength (`maxH_f`) | `Hrel = 17 m`; Eq. 3, p. 6 | “Hrel = 17 m representing the 100-year wave height” |
| New structure: smallest counted wave (`minHDe_f`) | No numerical value in text; see the new-structure resistance curve in Figure 3c, p. 7 | “For a variety of steel structures, different cyclic structural resistance curves would apply.” |
| New structure: strength (`maxHDe_f`) | No numerical value in text; see the new-structure resistance curve in Figure 3c, p. 7 | “two types of steel structures are considered” |

## End-of-Life and Network

| Dashboard parameter | Paper value / where to look | Paper quotation |
|---|---|---|
| Operation mode (`mode`, `eolMode`, `energyMode`) | Extension, decommissioning, or re-powering; Sections 4.2-4.3 and Figures 5-7, pp. 7-9 | “three options: extended operation, decommissioning, or re-powering” |
| End-of-life year (`eolYear`) | Example permit: 25 years after commissioning; combine with the 2015 simulation start to locate the example EOL around 2040; pp. 2 and 6 | “The permit shall expire 25 years after commissioning” |
| Decommissioning finished (`decomYear`) | No exact year in text; downtime is assumed. See decommissioning results in Figure 7, pp. 8-9 | “decommissioning and rebuilding take a certain amount of time” |
| Rebuild finished (`rebuildYear`) | No exact year in text; downtime is assumed. See decommissioning results in Figure 7, pp. 8-9 | “a certain downtime was considered between the decommissioning of the old structure” |
| Number of substations (`numOSS`) | Figure 4 shows three OWFs and three OSSs, p. 7 | “OSS (grey from 31 to 33)” |
| Turbines per substation (`numT`) | 60 turbines per OWF in simulations; Section 4.2, p. 7. Figure 4 uses 10 turbines per OWF for visualization. | “in the simulations, 60 turbines are employed per OWF” |
| Failure probability below limit (`plow`) | `p = 0.1`; Section 4.2, p. 8 | “the failure probability after EOL remains low, with p = 0.1” |
| Failure probability after limit (`phigh`) | `p = 0.5`; Section 4.2, p. 8 | “the failure probability increased to p = 0.5” |
| Damage wave threshold (`damage`, fixed) | No `Hmax` threshold specified in the paper text. | No corresponding threshold sentence in the paper. |

## Energy Output and Other Constants

| Dashboard parameter | Paper value / where to look | Paper quotation |
|---|---|---|
| Minimum/maximum turbine output (`minOut`, `maxOut`) | Baseline 4-6 MW; re-powering average 7.5 MW; decommissioning average 8.5 MW; Sections 4.2-4.3, pp. 7-9 | “an energy output between 4 and 6 MW per turbine”; “7.5 MW per turbine, on average”; “8.5 MW per turbine, on average” |
| Maximum-wave extra-height cap (`maxH`, fixed at 20 m) | Equation 2 gives the climate-dependent upper bound but does not state a 20 m cap; Section 4.1, p. 6 | “u = Hsig + Hsig · 0.07 · (y − ybase + 3)” |
| Repair duration | Two months; Section 4.2, p. 8 | “The duration of all needed repairs is set to two months” |
