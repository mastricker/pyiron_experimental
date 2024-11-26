import resistance_mdi
from pyiron_base import Project

import pandas as pd
import numpy as np

p = Project("./job_test/")
p.remove_jobs_silently()

job_measure = p.create_job(resistance_mdi.ResistanceGP, "job_gp")
print(job_measure.input)

# define already-measured data for dummy device
filename = "../notebooks/Ir-Pd-Pt-Rh-Ru_dataset.csv"
data = pd.read_csv(filename)

features = ["Ir", "Pd", "Pt", "Rh", "Ru"]
target = ["Resistance"]

data[target] = np.log(data[target])

max_iter = 100

params = {"df": data, "features": features, "target": target, "max_iter": max_iter}

job_measure.input.df = data
job_measure.input.features = features
job_measure.input.max_gp_iterations = max_iter
job_measure.input.target = target

# Output raw data

print(job_measure.input)

job_measure.run()

print("DEVICE, raw data = \n", job_measure.device.df)

print("Outside ", job_measure.output.mu_vs_iterations)

# print(job_measure.output["measurement_indices"])
# print(job_measure.output["resistance_prediction"])

# job_postproc = p.create_job(mdi_suite.xrd_postproc, "job_postproc")
# job_postproc.input.xrd_measurement = job_measure
# job_postproc.run()

# Note: this here needs to be put into a notebook with visualization
