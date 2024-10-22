from pyiron_base._tests import TestWithCleanProject
from pyiron_experimental.resistance_mdi import ResistanceGP

import pandas as pd
import numpy as np


class TestMDIResistance(TestWithCleanProject):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data_filename = "../notebooks/Ir-Pd-Pt-Rh-Ru_dataset.csv"
        cls.df = pd.read_csv(cls.data_filename)
        cls.reference_measurement_indices = [
            5,
            157,
            338,
            177,
            188,
            39,
            38,
            279,
            0,
            82,
            323,
            10,
            183,
            324,
            74,
            220,
        ]

    def setUp(self):
        self.job_name = "resistance_test"
        self.job = self.project.create.job.ResistanceGP(self.job_name)
        self.job.input.df = self.df
        self.job.input.df["Resistance"] = self.job.input.df["Resistance"]
        self.job.input.features = ["Ir", "Pd", "Pt", "Rh", "Ru"]
        self.job.input.target = ["Resistance"]
        self.job.input.max_gp_iterations = 11
        self.job.input.initialization_indices = [5, 157, 338, 177, 188]

    def test_static_workflow(self):
        self.job.run()

        measurement_indices = self.job.output["measurement_indices"]

        for value, reference in zip(
            measurement_indices, self.reference_measurement_indices
        ):
            self.assertEqual(value, reference)

    def test_load_static_workflow(self):
        self.test_static_workflow()
        job_load = self.project.load(self.job_name)

        measurement_indices = job_load.output["measurement_indices"]

        for value, reference in zip(
            measurement_indices, self.reference_measurement_indices
        ):
            self.assertEqual(value, reference)
