"""Date encoder 2.0 for HTM, ported to Python from the C++ implementation."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Union

import pandas as pd

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


@dataclass
class DateEncoderParameters:
    """Parameters for DateEncoder."""

    # Season: day of year (0..366), default radius 91.5 days (~4 seasons)
    season_width: int = 0
    season_radius: float = 91.5

    # Day of week: Monday=0, Tuesday=1, ... (C++ maps from tm_wday)
    day_of_week_width: int = 0
    day_of_week_radius: float = 1.0

    # Weekend flag (0/1, Fri 6pm through Sun midnight)
    weekend_width: int = 0

    # Holiday: boolean-ish with ramp, default dates = [[12, 25]] (month, day)
    holiday_width: int = 0
    holiday_dates: List[List[int]] = field(default_factory=lambda: [[12, 25]])

    # Time of day: 0..24 hours
    time_of_day_width: int = 0
    time_of_day_radius: float = 4.0

    # Custom day groups (e.g. ["mon,wed,fri"])
    custom_width: int = 0
    custom_days: List[str] = field(default_factory=list)

    # Verbose logging
    verbose: bool = False


class DateEncoder(BaseEncoder):
    """
    Python port of the HTM DateEncoder, using the existing ScalarEncoder + SDR.
    Encodes up to 6 attributes of a timestamp into one SDR:

      - season       (day-of-year)
      - dayOfWeek
      - weekend
      - customDays
      - holiday
      - timeOfDay
    """

    SEASON = 0
    DAYOFWEEK = 1
    WEEKEND = 2
    CUSTOM = 3
    HOLIDAY = 4
    TIMEOFDAY = 5

    def __init__(
        self, parameters: DateEncoderParameters, dimensions: Optional[List[int]] = None
    ) -> None:
        """Initialise all scalar sub-encoders and supporting metadata."""
        super().__init__(dimensions)
        self.parameters = copy.deepcopy(parameters)

        # encoders
        self.seasonEncoder: ScalarEncoder | None = None
        self.dayOfWeekEncoder: ScalarEncoder | None = None
        self.weekendEncoder: ScalarEncoder | None = None
        self.customDaysEncoder: ScalarEncoder | None = None
        self.holidayEncoder: ScalarEncoder | None = None
        self.timeOfDayEncoder: ScalarEncoder | None = None

        # for custom days mapping
        self.customDays_: Set[int] = set()

        # bucket index → position in buckets_
        self.bucketMap: Dict[int, int] = {}
        self.buckets_: List[float] = []
        # public alias like C++: const std::vector<Real64> &buckets = buckets_;
        self.buckets = self.buckets_

        self._size: int = 0  # total size of all sub-encoders
        self._initialize(parameters)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        """Emit verbose diagnostic messages when verbose mode is enabled."""
        if self.parameters.verbose:
            print("[DateEncoder] " + msg)

    @property
    def size(self) -> int:
        """Total number of bits produced by the configured sub-encoders."""
        return self._size

    # ------------------------------------------------------------------ #
    # Initialization (mirrors C++ initialize())
    # ------------------------------------------------------------------ #

    def _initialize(self, parameters: DateEncoderParameters) -> None:
        """Configure scalar encoders according to the supplied parameters."""
        args = parameters
        size = 0
        self.bucketMap.clear()
        self.buckets_.clear()

        # -------- Season --------
        if args.season_width != 0:
            p = ScalarEncoderParameters(
                minimum=0.0,
                maximum=366.0,
                clip_input=False,
                periodic=True,
                category=False,
                active_bits=args.season_width,
                sparsity=0.0,
                size=0,
                radius=args.season_radius,
                resolution=0.0,
                size_or_radius_or_category_or_resolution=0,
                active_bits_or_sparsity=0,
            )
            self.seasonEncoder = ScalarEncoder(p, [1])
            self.bucketMap[self.SEASON] = len(self.buckets_)
            self.buckets_.append(0.0)
            size += self.seasonEncoder._size
            self._log(
                f"Season encoder: buckets ~ {(p.maximum - p.minimum) / self.seasonEncoder._radius}, "
                f"active bits {self.seasonEncoder._active_bits}, width {self.seasonEncoder._size}"
            )

        # -------- Day of week --------
        if args.day_of_week_width != 0:
            p = ScalarEncoderParameters(
                minimum=0.0,
                maximum=7.0,
                clip_input=False,
                periodic=True,
                category=False,
                active_bits=args.day_of_week_width,
                sparsity=0.0,
                size=0,
                radius=args.day_of_week_radius,
                resolution=0.0,
                size_or_radius_or_category_or_resolution=0,
                active_bits_or_sparsity=0,
            )
            self.dayOfWeekEncoder = ScalarEncoder(p, [1])
            self.bucketMap[self.DAYOFWEEK] = len(self.buckets_)
            self.buckets_.append(0.0)
            size += self.dayOfWeekEncoder._size
            self._log(
                f"DayOfWeek encoder: buckets ~ {(p.maximum - p.minimum) / self.dayOfWeekEncoder._radius}, "
                f"active bits {self.dayOfWeekEncoder._active_bits}, width {self.dayOfWeekEncoder._size}"
            )

        # -------- Weekend --------
        if args.weekend_width != 0:
            p = ScalarEncoderParameters(
                minimum=0.0,
                maximum=1.0,
                clip_input=False,
                periodic=False,
                category=True,  # binary category 0/1
                active_bits=args.weekend_width,
                sparsity=0.0,
                size=0,
                radius=0.0,
                resolution=0.0,
                size_or_radius_or_category_or_resolution=0,
                active_bits_or_sparsity=0,
            )
            self.weekendEncoder = ScalarEncoder(p, [1])
            self.bucketMap[self.WEEKEND] = len(self.buckets_)
            self.buckets_.append(0.0)
            size += self.weekendEncoder._size
            self._log(
                f"Weekend encoder: categories 2, active bits {self.weekendEncoder._active_bits}, "
                f"width {self.weekendEncoder._size}"
            )

        # -------- Custom days --------
        if args.custom_width != 0:
            if not args.custom_days:
                raise ValueError(
                    "DateEncoder: custom_days must contain at least one pattern string."
                )

            # Map strings to Python tm_wday (0=Mon..6=Sun)
            daymap = {
                "mon": 0,
                "tue": 1,
                "wed": 2,
                "thu": 3,
                "fri": 4,
                "sat": 5,
                "sun": 6,
            }
            for spec in args.custom_days:
                s = spec.lower()
                parts = [x.strip() for x in s.split(",") if x.strip()]
                for day in parts:
                    if len(day) < 3:
                        raise ValueError(f"DateEncoder custom_days parse error near '{day}'")
                    key = day[:3]
                    if key not in daymap:
                        raise ValueError(f"DateEncoder custom_days parse error near '{day}'")
                    self.customDays_.add(daymap[key])

            p = ScalarEncoderParameters(
                minimum=0.0,
                maximum=1.0,
                clip_input=False,
                periodic=False,
                category=True,  # boolean category
                active_bits=args.custom_width,
                sparsity=0.0,
                size=0,
                radius=0.0,
                resolution=0.0,
                size_or_radius_or_category_or_resolution=0,
                active_bits_or_sparsity=0,
            )
            self.customDaysEncoder = ScalarEncoder(p, [1])
            self.bucketMap[self.CUSTOM] = len(self.buckets_)
            self.buckets_.append(0.0)
            size += self.customDaysEncoder._size
            self._log(
                f"CustomDays encoder: boolean, active bits {self.customDaysEncoder._active_bits}, "
                f"width {self.customDaysEncoder._size}"
            )

        # -------- Holiday --------
        if args.holiday_width != 0:
            for day in args.holiday_dates:
                if len(day) not in (2, 3):
                    raise ValueError(
                        "DateEncoder: holiday_dates entries must be [mon,day] or [year,mon,day]."
                    )

            p = ScalarEncoderParameters(
                minimum=0.0,
                maximum=2.0,
                clip_input=False,
                periodic=True,
                category=False,
                active_bits=args.holiday_width,
                sparsity=0.0,
                size=0,
                radius=1.0,
                resolution=0.0,
                size_or_radius_or_category_or_resolution=0,
                active_bits_or_sparsity=0,
            )
            self.holidayEncoder = ScalarEncoder(p, [1])
            self.bucketMap[self.HOLIDAY] = len(self.buckets_)
            self.buckets_.append(0.0)
            size += self.holidayEncoder._size
            self._log(
                f"Holiday encoder: buckets ~ {(p.maximum - p.minimum) / self.holidayEncoder._radius}, "
                f"active bits {self.holidayEncoder._active_bits}, width {self.holidayEncoder._size}"
            )

        # -------- Time of day --------
        if args.time_of_day_width != 0:
            p = ScalarEncoderParameters(
                minimum=0.0,
                maximum=24.0,
                clip_input=False,
                periodic=True,
                category=False,
                active_bits=args.time_of_day_width,
                sparsity=0.0,
                size=0,
                radius=args.time_of_day_radius,
                resolution=0.0,
                size_or_radius_or_category_or_resolution=0,
                active_bits_or_sparsity=0,
            )
            self.timeOfDayEncoder = ScalarEncoder(p, [1])
            self.bucketMap[self.TIMEOFDAY] = len(self.buckets_)
            self.buckets_.append(0.0)
            size += self.timeOfDayEncoder._size
            self._log(
                f"TimeOfDay encoder: buckets ~ {(p.maximum - p.minimum) / self.timeOfDayEncoder._radius}, "
                f"active bits {self.timeOfDayEncoder._active_bits}, width {self.timeOfDayEncoder._size}"
            )

        if size == 0:
            raise ValueError("DateEncoder: No widths were provided; nothing to encode.")

        self._size = size

    # ------------------------------------------------------------------ #
    # Public encode API (similar to C++ overloads)
    # ------------------------------------------------------------------ #

    def encode(self, input_value: datetime | pd.Timestamp, output: SDR) -> None:
        """
        Encode a timestamp-like value into `output` SDR.

        input_value:
          - None          -> current local time
          - int/float     -> UNIX epoch seconds
          - datetime      -> datetime (naive treated as local)
          - struct_time   -> used directly
        """
        if output.size != self.size:
            raise ValueError(f"Output SDR size {output.size} != DateEncoder size {self.size}")

        if input_value is None:
            t = time.localtime()
        elif isinstance(input_value, (int, float)):
            t = time.localtime(float(input_value))
        elif isinstance(input_value, datetime):
            ts = input_value.timestamp()
            t = time.localtime(ts)
        elif isinstance(input_value, time.struct_time):
            t = input_value
        else:
            raise TypeError(f"Unsupported type for DateEncoder.encode: {type(input_value)}")

        # Collect per-attribute SDRs to later concatenate
        sdrs: List[SDR] = []

        # verbose
        self._log(f"Encoding {time.asctime(t)} {'(dst)' if t.tm_isdst > 0 else ''}")

        # --- Season: day of year (0-based) ---
        if self.seasonEncoder is not None:
            day_of_year = float(t.tm_yday - 1)  # tm_yday is 1..366
            s = SDR(dimensions=[self.seasonEncoder._size])
            self.seasonEncoder.encode(day_of_year, s)
            # bucket index: floor(day / radius)
            bucket_idx = math.floor(day_of_year / self.seasonEncoder._radius)
            self.buckets_[self.bucketMap[self.SEASON]] = float(bucket_idx)
            self._log(f"  season: {day_of_year} -> bucket {bucket_idx}")
            sdrs.append(s)

        # --- Day of week (Monday=0..Sunday=6, same as header comment) ---
        if self.dayOfWeekEncoder is not None:
            # C++: dayOfWeek = (tm_wday + 6) % 7, with tm_wday 0=Sun..6=Sat
            # Python tm_wday: 0=Mon..6=Sun
            # So emulate C++ tm_wday first:
            c_tm_wday = (t.tm_wday + 1) % 7  # now 0=Sun..6=Sat
            day_of_week = float((c_tm_wday + 6) % 7)
            s = SDR(dimensions=[self.dayOfWeekEncoder._size])
            self.dayOfWeekEncoder.encode(day_of_week, s)
            radius = max(self.dayOfWeekEncoder._radius, 1e-9)
            bucket_val = day_of_week - math.fmod(day_of_week, radius)
            self.buckets_[self.bucketMap[self.DAYOFWEEK]] = bucket_val
            self._log(f"  dayOfWeek: {day_of_week} -> bucket start {bucket_val}")
            sdrs.append(s)
        else:
            # still compute c_tm_wday for weekend/custom use
            c_tm_wday = (t.tm_wday + 1) % 7

        # --- Weekend flag (Fri 18:00 .. Sun 23:59) ---
        if self.weekendEncoder is not None:
            # C++ logic uses C tm_wday (0=Sun..6=Sat)
            if c_tm_wday == 0 or c_tm_wday == 6 or (c_tm_wday == 5 and t.tm_hour > 18):
                val = 1.0
            else:
                val = 0.0
            s = SDR(dimensions=[self.weekendEncoder._size])
            self.weekendEncoder.encode(val, s)
            self.buckets_[self.bucketMap[self.WEEKEND]] = val
            self._log(f"  weekend: {val}")
            sdrs.append(s)

        # --- Custom days ---
        if self.customDaysEncoder is not None:
            # customDays_ holds Python tm_wday (0=Mon..6=Sun)
            custom_val = 1.0 if t.tm_wday in self.customDays_ else 0.0
            s = SDR(dimensions=[self.customDaysEncoder._size])
            self.customDaysEncoder.encode(custom_val, s)
            self.buckets_[self.bucketMap[self.CUSTOM]] = custom_val
            self._log(f"  customDay: {custom_val}")
            sdrs.append(s)

        # --- Holiday ramp ---
        if self.holidayEncoder is not None:
            val = self._holiday_value(t)
            s = SDR(dimensions=[self.holidayEncoder._size])
            self.holidayEncoder.encode(val, s)
            self.buckets_[self.bucketMap[self.HOLIDAY]] = math.floor(val)
            self._log(f"  holiday: {val}")
            sdrs.append(s)

        # --- Time of day ---
        if self.timeOfDayEncoder is not None:
            tod = t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0
            s = SDR(dimensions=[self.timeOfDayEncoder._size])
            self.timeOfDayEncoder.encode(tod, s)
            radius = max(self.timeOfDayEncoder._radius, 1e-9)
            bucket_val = tod - math.fmod(tod, radius)
            self.buckets_[self.bucketMap[self.TIMEOFDAY]] = bucket_val
            self._log(f"  timeOfDay: {tod} -> bucket start {bucket_val}")
            sdrs.append(s)

        if not sdrs:
            raise RuntimeError("DateEncoder misconfigured: no sub-encoders enabled.")

        # Concatenate SDRs into `output`
        all_sparse: List[int] = []
        offset = 0
        for s in sdrs:
            for idx in s.get_sparse():
                all_sparse.append(idx + offset)
            offset += s.size

        output.zero()
        output.set_sparse(all_sparse)
        self._log(f"  result: size {output.size}, active {len(all_sparse)}")

    # ------------------------------------------------------------------ #
    # Holiday helper (matches C++ logic)
    # ------------------------------------------------------------------ #

    def _holiday_value(self, t: time.struct_time) -> float:
        """Return the holiday ramp value for the provided timestamp."""
        seconds_per_day = 86400.0
        input_ts = time.mktime(t)

        for h in self.parameters.holiday_dates:
            if len(h) == 3:
                year, mon, day = h
            else:
                year = t.tm_year
                mon, day = h
            h_ts = self.mktime(year, mon, day)

            if input_ts > h_ts:
                diff = input_ts - h_ts
                if diff < seconds_per_day:
                    return 1.0
                elif diff < 2.0 * seconds_per_day:
                    return 1.0 + (diff - seconds_per_day) / seconds_per_day
            else:
                diff = h_ts - input_ts
                if diff < seconds_per_day:
                    return 1.0 - diff / seconds_per_day

        return 0.0

    @staticmethod
    def mktime(year: int, mon: int, day: int, hr: int = 0, minute: int = 0, sec: int = 0) -> float:
        """Convenience to generate unix epoch seconds like the C++ static mktime."""
        dt = datetime(year, mon, day, hr, minute, sec)
        return time.mktime(dt.timetuple())


if __name__ == "__main__":
    params = DateEncoderParameters(
        season_width=10,
        day_of_week_width=5,
        weekend_width=3,
        holiday_width=4,
        time_of_day_width=6,
        custom_width=3,
        custom_days=["mon,wed,fri"],
        verbose=True,
    )
    encoder = DateEncoder(params)
    output = SDR(dimensions=[encoder.size])
    encoder.encode(datetime.now(), output)
    print("Output size:", output.size)
    print("Active indices:", output.get_sparse())
