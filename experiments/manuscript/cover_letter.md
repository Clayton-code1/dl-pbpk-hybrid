# Cover Letter — Journal of Pharmacokinetics and Pharmacodynamics

---

**[Date of submission]**

The Editor-in-Chief
*Journal of Pharmacokinetics and Pharmacodynamics*

---

Dear Editor,

We are pleased to submit the manuscript titled **"A Hybrid Graph Neural Network and Mechanistically Constrained Pharmacokinetic Framework for Multi-Drug Plasma Concentration Prediction, Validated Against Real Human Data"** by Clayton Takayidza and Maronge Musara (Harare Institute of Technology, Zimbabwe) for consideration as an Original Paper in the *Journal of Pharmacokinetics and Pharmacodynamics*.

**Scientific motivation.** Predicting individual plasma drug concentration profiles is central to rational dosing, yet classical population pharmacokinetic models require per-drug clinical parameterisation that is time-consuming and unavailable early in development. Machine-learning approaches offer flexibility but discard the known physics of drug absorption and elimination, limiting extrapolation and interpretability. Our work addresses this gap.

**What we did.** We developed a hybrid framework coupling a pretrained graph neural network (GNN) molecular encoder with a differentiable one-compartment oral PK ODE simulator. The model accepts a drug molecular graph plus five patient covariates (weight, dose, age, sex) and jointly predicts clearance per kilogram, volume of distribution per kilogram, and absorption rate constant in a single forward pass. These parameters are passed to the ODE to produce the full plasma concentration–time profile. The complete architecture has 94,403 parameters and is trained end-to-end.

**Key findings.** On simulated benchmark data for six chemically diverse drugs, the hybrid model achieved a mean R² of 0.851 across all drugs, compared with 0.341 for a realistic population PK baseline and 0.815 for a GNN without mechanistic constraints. An ablation study quantified the independent contribution of the mechanistic ODE, the pretrained encoder, and encoder fine-tuning. Zero-shot generalisation to a withheld seventh drug (ibuprofen) yielded R² = 0.836. Monte Carlo uncertainty bands achieved 87.5% observed coverage at the nominal 90% level. Critically, we validated the model on real human data using forward-only inference (no retraining): R² = 0.673 on the R Theoph dataset (12 subjects, 132 observations) and R² = 0.668 on the warfarin absorption-present subgroup (13 subjects) of the nlmixr2data warfarin dataset—under a deliberately stressful 20× training-dose extrapolation challenge. We report these simulation-to-reality gaps explicitly as informative quantitative characterisations of model limitations rather than suppressing them.

**Why JPKPD.** The Journal of Pharmacokinetics and Pharmacodynamics is the natural home for this work: it combines a novel mechanistic modelling architecture with quantitative pharmacokinetic evaluation across multiple drugs, real-data validation, and honest assessment of where simulation-trained models succeed and fail. The paper speaks directly to the journal's readership in pharmacometrics, computational PK/PD, and quantitative systems pharmacology.

**Originality and overlap.** This manuscript has not been published previously and is not under consideration elsewhere. All code, training scripts, and results are publicly available at https://github.com/Clayton-code1/dl-pbpk-hybrid under the MIT Licence, in accordance with the journal's open-science principles.

We have no conflicts of interest to declare and no special handling requests.

We look forward to hearing from you.

Yours sincerely,

**Clayton Takayidza**
Harare Institute of Technology, Harare, Zimbabwe
Email: takayidzaclayton@gmail.com

**Maronge Musara**
Harare Institute of Technology, Harare, Zimbabwe
