"""Errors separate identification, symbolic support and numerical support."""


class TCBError(Exception):
    pass


class NotTransportableError(TCBError):
    pass


class UnsupportedFormulaError(TCBError):
    pass


class MissingDistributionError(TCBError):
    pass


class PositivityError(TCBError):
    pass


class ApproximationWarning(UserWarning):
    pass
