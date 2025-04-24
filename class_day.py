import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import datetime


class DayPattern:
    def __init__(self, date=datetime.date.today(), day_type=0):
        self.date = date
        self.day = date.day
        self.week_number = date.isocalendar()[1]
        self.year = self.date.year
        self.month = self.date.month
        self.week_day = self.date.weekday()

        self.day_type = day_type
        self.number_high_picks = 1
        self.number_low_picks = self.number_high_picks + 1
        self.max_flow = 1
        self.min_flow = 0
        # self.hourly_time_range = self.create_hourly_time_rage(self.day, self.week_number)
        # self.hourly_time_range = self.create_hourly_time_rage(self.day, self.week_number)

    def update_day(self, day):
        self.day = day

    def update_day_type(self, day_type):
        self.day_type = day_type


    def update_max_flow(self, max_flow):
        self.max_flow = max_flow

    def update_min_flow(self, min_flow):
        self.min_flow = min_flow

    def update_num_picks(self, num_high_picks):
        self.number_high_picks = num_high_picks
        self.number_low_picks = num_high_picks + 1

    def create_daily_demand_pattern(self):
        pass

    def create_24_hours_rage(self):
        return range(0, 24)


if __name__ == "__main__":
    print(datetime.date.today())
    daypa_ttern = DayPattern()
    week = daypa_ttern.date
    print(dir(week))
