#!/bin/bash

sqlite3 -line barsave.db 'select system, ins, sum(pl) from backtest_result where abs(pl) > 0  group by system, ins;'
