import unittest
from smtm import TddExercise
from unittest.mock import *
import requests


class TddExerciseIntegrationTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialize_correctly(self):
        ex = TddExercise()
        ex.initialize_from_server("2020-02-25T06:41:00Z", 60)

        self.assertEqual(len(ex.data), 60)
        self.assertEqual(ex.data[0]["candle_date_time_utc"], "2020-02-25T05:41:00")
        self.assertEqual(ex.data[-1]["candle_date_time_utc"], "2020-02-25T06:40:00")
