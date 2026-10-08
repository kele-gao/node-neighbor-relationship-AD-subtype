# Simulated example data

The three CSVs contain 50 synthetic subjects (SIM_001 through SIM_050), not real participants. They share identical subject IDs and row order, with 400 ROI columns in GrayVol.csv.

Seed: 20261007. Generation is based on a 30-subject source subset. Group, sex and handedness proportions are approximately preserved; education is sampled within groups. Age, centiloids, eTIV and ROI volumes are jointly simulated using group/sex means and a residual covariance with 20% diagonal shrinkage, then bounded to source variable ranges. This approximates source distributions and correlations; it does not preserve them exactly or validate biological effects. Site labels, participant/image IDs and scan dates are newly generated.

Use only to demonstrate the analysis workflow. These files cannot reproduce manuscript estimates. Statistical synthesis does not provide a formal differential-privacy guarantee; source-data sharing conditions still apply.
