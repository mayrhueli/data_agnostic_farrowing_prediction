# Copyright (C) 2026 Elisabeth Mayrhuber
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Score a detected nest-building changepoint against the farrowing onset
time, using a 24h/48h tolerance-window scheme: a detection is "on time"
if it falls within the window before farrowing, "early" if it comes
earlier than that, and "late" if it falls after farrowing onset or
never fires at all.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DetectionResult:
    method: str
    signal: str
    changepoint: Optional[pd.Timestamp]
    farrowing_time: pd.Timestamp
    classification_24h: str
    classification_48h: str
    lead_time_hours: Optional[float]


def _classify(changepoint: Optional[pd.Timestamp], farrowing_time: pd.Timestamp, window_hours: float) -> str:
    if changepoint is None:
        return "late"  # "detected too late or not at all" 
    window_start = farrowing_time - pd.Timedelta(hours=window_hours)
    if changepoint < window_start:
        return "early"
    if changepoint > farrowing_time:
        return "late"
    return "on_time"


def evaluate_detection(method: str, signal: str, changepoint: Optional[pd.Timestamp],
                        farrowing_time: pd.Timestamp) -> DetectionResult:
    lead_time = None
    if changepoint is not None:
        lead_time = (farrowing_time - changepoint).total_seconds() / 3600.0

    return DetectionResult(
        method=method,
        signal=signal,
        changepoint=changepoint,
        farrowing_time=farrowing_time,
        classification_24h=_classify(changepoint, farrowing_time, 24),
        classification_48h=_classify(changepoint, farrowing_time, 48),
        lead_time_hours=lead_time,
    )
