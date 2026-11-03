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

"""NestDetect: a sliding-window nest-building detector, based on rolling
12h/24h averages of the activity signal.
"""
import pandas as pd

DAY = 1440
HALF_DAY = 720
TWO_DAYS = 2880

WINDOW_SIZE = 720
WINDOW_STEP = 180


def detect_nest_building(signal: pd.DataFrame):
    """Slide a 12h window (stepped every 3h) over a 1-minute
    ``timestamp, Acc`` series and return the first triggered alarm.

    Returns (changepoint_timestamp, alarm_label, signal_so_far, window_index)
    or (None, None, None, None) if no alarm fires before the data runs out.
    """
    window_index = 0
    signal_so_far = None
    while True:
        if window_index >= DAY:
            if window_index + WINDOW_SIZE > len(signal):
                break

            window = signal.iloc[window_index:window_index + WINDOW_SIZE]
            signal_so_far = window if signal_so_far is None else pd.concat([signal_so_far, window])

            avg_rolling_24 = (signal_so_far.iloc[:-DAY].Acc.rolling(window=DAY).mean()).mean()
            std_24 = (signal_so_far.iloc[:-DAY].Acc.rolling(window=DAY).std()).mean()
            last_24 = signal_so_far.Acc.tail(DAY).mean()

            rolling_12 = signal_so_far.Acc.rolling(window=HALF_DAY).mean()
            max_last_12 = rolling_12.tail(WINDOW_SIZE).max()
            max_12 = rolling_12.head(window_index).max()

            rolling_24 = signal_so_far.Acc.rolling(window=DAY).mean()
            max_last_24 = rolling_24.tail(WINDOW_SIZE).mean()
            max_24 = rolling_24.tail(WINDOW_SIZE).head(WINDOW_SIZE).mean()

            if (window_index + WINDOW_SIZE > DAY * 2.5) and max_last_12 > max_12 * 1.75:
                return window.iloc[0].timestamp, "Alarm 2", signal_so_far, window_index

            if (window_index + WINDOW_SIZE > DAY * 3.5) and last_24 >= (avg_rolling_24 + std_24) * 0.8:
                return window.iloc[0].timestamp, "Alarm 1", signal_so_far, window_index

            if (window_index + WINDOW_SIZE > DAY * 3.5) and max_last_24 > max_24:
                return window.iloc[0].timestamp, "Alarm 3", signal_so_far, window_index

        window_index += WINDOW_STEP
    return None, None, None, None
