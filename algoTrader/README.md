# Triangle Trader
WIll look for triangle formations on the 30D Timeframe and place STOP Orders to catch the breakouts

Use at your own risk!

There is no attempt to manage the trade after the STOP has been triggered. This has to be done with a different script

backtest is throttled to 1 Day/hr to not hurt my AWS computation time too much

controller.py: Implements the controller class which interfaces the Broker API and places orders generated from the indicator scripts in the market. There are further controls to ensure the orders align with current trend.

triangle.py: Looks to construct a consolidating triangle on the daily barchart. If such a triangle can be constructed STOP orders just outside the S/R levels will be passed to the controller.

triangleh4.py: Same as triangle but on H4 time frame. It is a generalization of the daily indicator and generalizes with regard to the time frame

triangleh4_lim: Similar to triangleh4 but places limit orders

sentiment.py: Attempts to act on the traders sentiment as seen on Oanda position book and myfxbook. The latter has proven to be very unreliable which makes this a pain to develop
