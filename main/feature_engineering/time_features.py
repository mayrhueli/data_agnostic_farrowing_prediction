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

from scipy.stats import linregress

import pandas as pd
from feature_engineering.constants import DAY, TWO_DAYS, HALF_DAY


def calculate_features_for_timeseries(series, eventtime, full_features=False):
    if series.empty:
        print("series is empty")
        features = {"mean_day": None,
                    "mean_night": None,
                    "mean_24": None,
                    "mean_48": None,
                    "mean_last_12": None,
                    "mean_last_24": None,
                    "trend_last_12": None,
                    "trend_last_24": None,
                    "crated": None,
                    "remaining_minutes": None
                    }

        features_df = pd.DataFrame([features])
        return features_df
    startdatetime = series.timestamp.min()

    # calculate mean between 22:00 and 03:00 for night mean
    start = pd.Timestamp((startdatetime + pd.DateOffset(days=1)).date()).replace(hour=22)
    end = pd.Timestamp((startdatetime + pd.DateOffset(days=2)).date()).replace(hour=4)
    mean_night = series[(start <= series.timestamp) & (series.timestamp <= end)].Acc.mean()

    # calculate mean between 10:00 and 18:00 for day mean
    start = pd.Timestamp((startdatetime + pd.DateOffset(days=1)).date()).replace(hour=10)
    end = pd.Timestamp((startdatetime + pd.DateOffset(days=1)).date()).replace(hour=18)
    mean_day = series[(start <= series.timestamp) & (series.timestamp <= end)].Acc.mean()

    # calculate mean for 24 h with guaranteed no nest building
    mean_24 = series.head(DAY).Acc.mean()

    # calculate mean for 48 h with guaranteed no nest building
    mean_48 = series.head(TWO_DAYS).Acc.mean()

    # calculate mean over the last 12h
    mean_last_12 = series.tail(HALF_DAY).Acc.mean()

    # calculate mean over last 24h
    mean_last_24 = series.tail(DAY).Acc.mean()

    # trend of the last 12h
    res_12 = linregress(series.tail(HALF_DAY).index, series.tail(HALF_DAY).Acc)
    trend_12 = res_12.slope

    # trend of the last 24h
    trend_24 = linregress(series.tail(DAY).index, series.tail(DAY).Acc).slope

    # 0 if sow is not crated 1 if sow is crated TODO
    crated = 0

    remaining_minutes = pd.Timedelta(eventtime - series.timestamp.max()).total_seconds() / 60

    features = {"mean_day": mean_day,
                "mean_night": mean_night,
                "mean_24": mean_24,
                "mean_48": mean_48,
                "mean_last_12": mean_last_12,
                "mean_last_24": mean_last_24,
                "trend_last_12": trend_12,
                "trend_last_24": trend_24,
                "crated": crated,
                "remaining_minutes": remaining_minutes
                }

    features_df = pd.DataFrame([features])

    df = None
    if full_features:
        import tsfel  # imported lazily: only this (unused by video_activity/) path needs it
        cfg_file = tsfel.get_features_by_domain(domain="statistical")
        extracted_features = tsfel.time_series_features_extractor(cfg_file, series.Acc, fs=None)
        df = pd.concat([features_df, extracted_features], axis=1)
    else:
        df = features_df

    return df
