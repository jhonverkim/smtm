import unittest
from smtm import SimulationTrader
from unittest.mock import *
import requests

class SimulationTraderTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialize_initialize_virtual_market(self):
        trader = SimulationTrader()
        class DummyHttp():
            pass
        http = DummyHttp()
        trader.market.initialize = MagicMock()
        trader.market.deposit = MagicMock()
        trader.initialize(http, "mango", 500, 5000)
        trader.market.initialize.assert_called_once_with(http, "mango", 500)
        trader.market.deposit.assert_called_once_with(5000)
        self.assertEqual(trader.is_initialized, True)

    def test_initialize_set_is_initialized_False_when_invalid_market(self):
        trader = SimulationTrader()
        class DummyHttp():
            pass
        http = DummyHttp()
        trader.market = "make exception"
        trader.initialize(http, "mango", 500, 5000)
        self.assertEqual(trader.is_initialized, False)

    def test_send_request_call_callback_with_result_of_market_handle_quest(self):
        trader = SimulationTrader()
        trader.is_initialized = True
        class DummyRequest():
            pass
        dummy_request = DummyRequest()
        dummy_request.id = "mango"
        dummy_request.type = "orange"
        callback = MagicMock()
        trader.market.send_request = MagicMock(return_value="banana")
        trader.send_request(dummy_request, callback)
        trader.market.send_request.assert_called_once_with(dummy_request)
        callback.assert_called_once_with("banana")

    def test_send_request_call_raise_exception_SystemError_when_is_initialized_False(self):
        trader = SimulationTrader()
        trader.is_initialized = False
        try:
            trader.send_request(None, None)
        except SystemError as msg:
            self.assertEqual(msg, ANY)

    def test_send_request_call_raise_exception_SystemError_when_market_is_invalid(self):
        trader = SimulationTrader()
        trader.is_initialized = True
        trader.market = "make exception"
        try:
            trader.send_request(None, None)
        except SystemError as msg:
            self.assertEqual(msg, ANY)

    def test_send_request_call_raise_exception_SystemError_when_callback_make_TypeError(self):
        trader = SimulationTrader()
        trader.is_initialized = True
        try:
            trader.send_request(None, None)
        except SystemError as msg:
            self.assertEqual(msg, ANY)
